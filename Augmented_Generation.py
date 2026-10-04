from Hybrid_Dual_Indexing import Keyword_Search, Semantic_Search
from PreRetrival_and_PostRetrieval import Context_Processer
from Retrieval import Retriever
from langchain_ollama import OllamaLLM
import json
import ast
import re
import os
import logging


class RAG:

    def __init__ (self, model = "qwen2.5:0.5b-instruct-q5_k_m", base_url = None, log_file_path = "Chat_History.log", retriever = None, log_interactions = None):

        self.log_file_path = log_file_path
        self.log_interactions = (log_interactions if log_interactions is not None else
                                 os.getenv ("ENABLE_CHAT_LOGGING", "false").lower () in ("1", "true", "yes", "on"))
        self.chat_history = "No prior conversation"
        self.reset_request_state ()

        base_url = base_url or os.getenv ("OLLAMA_BASE_URL") or "http://host.docker.internal:11434"
        self.llm = OllamaLLM (model = model, base_url = base_url)
        self.retriever = retriever if retriever is not None else Retriever ()
        self.context_processer = Context_Processer (model = model, base_url = base_url)

    def Summarize_Chat_History (self):
        if self.status != "success":
            return

        text = (f"- Previous Conversation | ORIGINAL USER QUERY: {self.user_query}. "
                f"REWRITTEN USER QUERY: {self.rewritten_query}. ANSWER: {self.final_response}.")

        prompt = f"""
            ### SYSTEM ROLE: You are an expert context-distiller. Your task is to summarize the "Current Chat History Summary" with the "New Interaction" between a User and an AI into a concise, information-dense technical brief (If the New Interaction contradicts or updates the Current Summary, prioritize the New Interaction).
            ### Objectives:
            1. Preserve Entities: Keep all specific names, tools, libraries, and technical constants.
            2. Maintain State: Clearly identify what has been resolved and what remains an open question.
            3. Eliminate Fluff: Remove greetings, apologies, and conversational fillers.
            4. Temporal Logic: If preferences changed during the chat, only record the most recent decision.

            ### Current Chat History Summary: {self.chat_history}
            ### New Interaction: {text}
            
            ### Output Format: [Provide a concise, 1-2 sentence summary of what was already discussed / attempted]
        """

        response = self.llm.invoke (prompt)
        self.chat_history = response.strip ()
    
    def Retrieval (self, do_keyword_search = True, do_semantic_search = True, do_graph_search = True,
                    do_RRF = True, do_cross_encoder = True,
                    do_query_expansion = False, do_rewrite = True, do_query_HyDE = False, replace_query_by_HyDE = False,
                    do_chunk_ordering = False, do_extractive_compression = False):

        understand, self.rewritten_query, self.intent = self.context_processer.user_query_understanding (self.user_query, self.chat_history, do_query_expansion, do_rewrite, do_query_HyDE, replace_query_by_HyDE)

        if self.intent == "RAG_SEARCH":
            top_k_chunks = self.retriever.hybrid_retrieval (understand, self.rewritten_query, do_keyword_search, do_semantic_search, do_RRF, do_cross_encoder)
            displayed_chunks = top_k_chunks if do_chunk_ordering else top_k_chunks[:5]
            self.retrieved_context = self.describe_retrieved_chunks (displayed_chunks)
            self.hybrid_text = self.context_processer.context_retrieval_processing (top_k_chunks, self.rewritten_query, do_chunk_ordering, do_extractive_compression)
            if do_graph_search:
                try:
                    entities_list = self.context_processer.entity_extraction (understand, do_graph_search)
                    self.graph_text = self.retriever.graph_retrieve (entities_list, self.rewritten_query, do_graph_search, do_cross_encoder)
                except Exception as error:
                    self.graph_text = ""
                    self.graph_warning = "Graph lookup was unavailable; this response uses retrieved passages only."
                    logging.getLogger (__name__).warning ("Graph lookup failed: %s", type (error).__name__)
            graph_text = self.graph_text if isinstance (self.graph_text, str) else "\n".join (self.graph_text or [])
            if graph_text.strip ():
                self.retrieved_context.append ({
                    "kind": "Graph relationships",
                    "source": "SympScan graph snapshot",
                    "disease": "",
                    "text": graph_text.strip (),
                })

    @staticmethod
    def describe_retrieved_chunks (chunks):

        context = []
        seen = set ()
        for chunk in chunks:
            text = chunk.page_content.strip ()
            if not text:
                continue
            metadata = getattr (chunk, "metadata", {}) or {}
            source = str (metadata.get ("source") or "Unknown source")
            disease = str (metadata.get ("disease_name") or "")
            identity = (source, disease, text)
            if identity in seen:
                continue
            seen.add (identity)
            context.append ({
                "kind": "Retrieved passage",
                "source": source,
                "disease": disease,
                "text": text,
            })
        return context

    def response_context_snapshot (self):
        return [item.copy () for item in self.retrieved_context]

    def Augmentation (self):

        if self.intent == "RAG_SEARCH":
            prompt = f""" 

            ### SYSTEM ROLE: You are a Medical Knowledge Engine. Your task is to synthesize an answer to the User Question (supported by Chat History) using two specific data streams:
            - Dataset graph relationships: {self.graph_text}
            - Retrieved dataset passages: {self.hybrid_text}

            ### USER QUESTION: {self.rewritten_query}
            ### CHAT HISTORY: {self.chat_history}

            ### CONSTRAINTS:
            - Use ONLY the provided sources.
            - Do not use phrases like "Based on the facts provided" or "According to the documents."
            - All values must be strings (no lists / arrays)

            ### OUTPUT REQUIREMENTS: Your response must be a valid JSON object
            ### REQUIRED JSON FORMAT (COPY EXACTLY):
            {{
            "answer": "concise 2-5 sentences",
            "disease": "Disease mentioned in the text",
            "medication": "Comma-separated of drugs mentioned",
            "advice": "Comma-separated list of precautions, dietary needs, exercises mentioned"
            }}
            ### EXAMPLES (FOLLOW THIS FORMAT):
            {{
            "answer": "Diabetes causes high blood sugar. Managed with medication and diet.",
            "disease": "Diabetes mellitus, Heart attack",
            "medication": "Metformin,  Atorvastatin",
            "advice": "Low sugar diet, daily walking"
            }}
            """

        else:
            prompt = f""" 

            ### SYSTEM ROLE: You are the "Medical Guide," a professional and empathetic assistant for a Medical Knowledge RAG system.
            ### TASK: Engage with the user's non-medical query (greetings, small talk, or app capabilities) while maintaining a clinical and helpful persona.

            ### GUIDELINES
            - Tone: Friendly but grounded. Avoid overly "bubbly" AI tropes.
            - Redirection: Always gently remind the user that you are specialized in symptom analysis and disease name based on the Symptom Scan database.
            - Conciseness: Keep responses under 3 sentences unless explaining a feature.

            ### USER QUESTION: {self.rewritten_query}
            ### CHAT HISTORY: {self.chat_history}

            ### OUTPUT REQUIREMENTS: Your response must be a valid JSON object and no list-format output
            ### JSON SCHEMA:
            {{
            "chitchat": "brief and redirect 2-5 sentences."
            }}
            ### EXAMPLES (FOLLOW THIS FORMAT):
            {{
            "chitchat": "The sky is blue, I like it !"
            }}
            """

        return prompt

    def Prompt_Fixed_after_Fail_Format_Check (self, fail_response, fail_times):

        if self.intent == "RAG_SEARCH":
            if fail_times == 1:
                fix_instruction = f"""
                    ### FAILED ATTEMPT FIX INSTRUCTIONS:
                        1. Ensure the response is a single, valid JSON object.
                        2. Remove markdown code blocks like ```json.
                        3. No text before or after the JSON.
                        4. If a value is empty, use "".
                        5. Ensure all quotes are double quotes (").
                        6. Do not add any conversational text.
                        7. Ensure all keys ("answer", "disease", "medication", "advice") are present.
                """
            else:
                fix_instruction = ""
            
            prompt = fix_instruction + f"""
                ### RECTIFICATION NOTICE {fail_times}: Your previous responses failed the JSON schema check. 
                ### PREVIOUS FAILED ATTEMPT {fail_times}: {fail_response}
            """
        
        else:
            if fail_times == 1:
                fix_instruction = f"""
                    ### FAILED ATTEMPT FIX INSTRUCTIONS:
                        1. Ensure the response is a single, valid JSON object.
                        2. Remove markdown code blocks like ```json.
                        3. No text before or after the JSON.
                        4. If a value is empty, use "".
                        5. Ensure all quotes are double quotes (").
                        6. Do not add any conversational text.
                        7. Ensure the key ("chitchat") are present.
                """
            else:
                fix_instruction = ""

            prompt = fix_instruction + f"""
                ### RECTIFICATION NOTICE {fail_times}: Your previous responses failed the JSON schema check. 
                ### PREVIOUS FAILED ATTEMPT {fail_times}: {fail_response}
            """ 
        
        return prompt        

    def Generation (self, prompt, format_fail = 3, content_fail = 1):

        first_time_response = ""
        format_fail_i = 0
        content_fail_j = 0

        while format_fail_i < format_fail and content_fail_j < content_fail:
            response_before_check = self.llm.invoke (prompt)
            if first_time_response == "":
                first_time_response = response_before_check

            response = self.response_format_check (response_before_check)
            if response == "":
                format_fail_i += 1
                prompt += self.Prompt_Fixed_after_Fail_Format_Check (response_before_check, format_fail_i)
            else:
                break

        if format_fail_i >= format_fail or content_fail_j >= content_fail:
            response = ("### ⚠️ System Note\n"
                        "**I'm sorry, but I couldn't find a specific medical response for that query.**\n\n"
                        "--- \n"
                        "**Possible reasons:**\n"
                        "* The query was too vague.\n"
                        "* The topic falls outside my medical knowledge base.\n"
                        "* The system encountered a retrieval error.\n\n"
                        "👉 *Try rephrasing your question with more specific symptoms.*")
            return first_time_response, response, "fail"

        if self.intent == "RAG_SEARCH":
            answer = response.get ("answer") or "The system doesn't have an answer for yours"
            disease = response.get ("disease") or "No disease determined"
            medication = response.get ("medication") or "No medication indicated"
            advice = response.get ("advice") or "No other advice"

            answer = self.process_valid_response (answer)
            disease = self.process_valid_response (disease)
            medication = self.process_valid_response (medication)
            advice = self.process_valid_response (advice)

            if answer == "Concise 2-5 sentences:":
                answer = "Please rely on other supporting detail."

            response = ( f"### 🩺 Medical Assessment\n"
                        f"{answer}\n\n"
                        f"--- \n"
                        f"**🏥 Condition:** {disease}  \n"
                        f"**💊 Medication:** {medication}  \n\n"
                        f"**💡 Pro-Tip:** \n"
                        f"> {advice}")

        else:
            response = response.get ("chitchat") or "I'm sorry, I'm having trouble processing that right now. May you ask specifically about medical stuffs, healthcare, ... ?"
            response = self.process_valid_response (response)
            response = (f"### 💬 Assistant Response\n"
                        f"{response}\n\n"
                        f"--- \n"
                        f"👉 *Ask me about symptoms, treatments, or healthcare advice for more detailed analysis.*")

        return first_time_response, response, "success"

    def process_valid_response (self, response):
        if isinstance (response, str):
            try:
                response = ast.literal_eval (response)
            except (SyntaxError, ValueError):
                pass

        def flatten (value):
            if isinstance (value, dict):
                for item in value.values ():
                    yield from flatten (item)
            elif isinstance (value, (list, tuple)):
                for item in value:
                    yield from flatten (item)
            elif value is not None:
                text = str (value).strip ()
                if text:
                    yield text

        response_text = ", ".join (flatten (response))
        if response_text and response_text[-1] not in ".?!":
            response_text += "."

        return response_text

    def response_format_check (self, response):

        required_keys = ("answer", "disease", "medication", "advice") if self.intent == "RAG_SEARCH" else ("chitchat",)
        if not isinstance (response, str):
            return ""

        decoder = json.JSONDecoder ()
        for match in re.finditer (r'\{', response):
            try:
                candidate, _ = decoder.raw_decode (response, match.start ())
            except json.JSONDecodeError:
                continue
            if isinstance (candidate, dict) and all (
                isinstance (candidate.get (key), str) for key in required_keys
            ):
                return candidate

        return ""

    def response_content_check (self, response, query, relevance_threshold = 0.0):

        prompt = f"""
            ### SYSTEM ROLE: Role: You are an expert grader and subject matter expert in Medical Field
            
            ### Task: Evaluate the provided Answer based on the Original Question
            
            ### User Question: {query}
            ### Answer to Evaluate: {response}

            ### Evaluation Criteria:
            1. Accuracy: Is the information factually correct? Identify any hallucinations or errors.
            2. Completeness: Does it address all parts of the question?
            3. Clarity & Flow: Is the explanation easy to follow and well-structured?
            4. Tone: Is the persona/voice appropriate for the intended audience?

            ### Output Format: Return ONLY a single floating-point number between 0.0 and 1.0 
        """

        score = self.llm.invoke (prompt)
        score_match = re.search(r"[-+]?\d*\.\d+|\d+", score.strip ())

        if not score_match:
            return 0.0, False

        score = float (score_match.group ())
        if score < relevance_threshold:
            return score, False
        
        return score, True
    
    def how_retrieval_helpful (self):

        if not self.hybrid_text and not self.graph_text:
            return 0.0

        prompt = f"""
            ### SYSTEM ROLE: You are a Retrieval Quality Auditor for a Medical RAG System.
            
            ### TASK: 
            Evaluate the usefulness of the RETRIEVED DOCUMENTS in answering the USER QUERY. 
            You must determine if the context provided enough "Evidence" to formulate a safe and accurate medical response.

            ### INPUT DATA:
            - USER QUERY: {self.user_query}
            - VECTOR/HYBRID CONTEXT: {self.hybrid_text if self.hybrid_text else "NOT PROVIDED"}
            - KNOWLEDGE GRAPH CONTEXT: {self.graph_text if self.graph_text else "NOT PROVIDED"}

            ### EVALUATION CRITERIA:
            1. RELEVANCE: Do the documents discuss the specific diseases or symptoms mentioned?
            2. SUFFICIENCY: Is there enough information to suggest medications, precautions, or advice as requested in the JSON schema?
            3. ACCURACY POTENTIAL: Will using this context prevent the LLM from hallucinating?
            4. SYNERGY: If both Graph and Hybrid data are present, do they complement each other?

            ### OUTPUT FORMAT:
            Return ONLY a single floating-point number between 0.0 and 1.0 representing the "Helpfulness Score."
            - 1.0: Perfect context; contains specific diagnosis, meds, and advice.
            - 0.5: Partial context; mentions the disease but lacks treatment or precaution details.
            - 0.0: Irrelevant; documents do not help answer the query at all.
        """
    
        score = self.llm.invoke (prompt)
        score_match = re.search(r"[-+]?\d*\.\d+|\d+", score.strip ())

        if not score_match:
            return 0.0

        score = float (score_match.group ())
        return score
    
    def Logging (self):

        if not self.log_interactions:
            return

        response_score = None
        retrieval_score = None
        if self.status == "success":
            response_score = self.response_content_check (self.final_response, self.user_query)[0]
            retrieval_score = self.how_retrieval_helpful ()

        each_line = {"RAW USER QUERY": self.user_query,
                    "REWRITTEN USER QUERY": self.rewritten_query,
                    "HYBRID RETRIEVAL": self.hybrid_text,
                    "GRAPH RETRIEVAL": self.graph_text,
                    "GRAPH WARNING": self.graph_warning,
                    "FIRST RESPONSE": self.first_response,
                    "FINAL RESPONSE": self.final_response,
                    "RESPONSE CONFIDENCE (0-1)": response_score,
                    "RETRIEVAL CONFIDENCE (0-1)": retrieval_score,
                    "STATUS": self.status
                    }
        each_line = json.dumps (each_line)

        with open (self.log_file_path, "a") as file:
            file.write (each_line + "\n")

    def Caching (self):

        pass

    def reset_request_state (self):

        self.user_query = ""
        self.rewritten_query = ""
        self.first_response = ""
        self.final_response = ""
        self.status = ""
        self.hybrid_text = ""
        self.graph_text = ""
        self.intent = ""
        self.retrieved_context = []
        self.graph_warning = ""

    def has_retrieved_evidence (self):

        def has_content (value):
            if isinstance (value, str):
                return bool (value.strip ())
            if isinstance (value, (list, tuple)):
                return any (has_content (item) for item in value)
            return bool (value)

        return has_content (self.hybrid_text) or has_content (self.graph_text)
    
    def RAG_PostOnline_Phase (self):

        self.Summarize_Chat_History ()
        self.Caching ()
        self.Logging ()

    def RAG_Online_Phase (self, user_query):

        self.reset_request_state ()
        self.user_query = user_query
        self.Retrieval ()
        if self.intent == "RAG_SEARCH" and not self.has_retrieved_evidence ():
            self.status = "no_evidence"
            self.final_response = (
                "I couldn't find supporting information for this question in the available data. "
                "Please ask a more specific question or consult a qualified healthcare professional."
            )
            return self.final_response

        prompt = self.Augmentation ()
        self.first_response, self.final_response, self.status = self.Generation (prompt)

        return self.final_response
