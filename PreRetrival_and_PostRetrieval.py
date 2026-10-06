from langchain_ollama import OllamaLLM
from runtime_config import ollama_settings
import re

class Context_Processer:

    def __init__ (self, model = None, base_url = None):

        model, base_url = ollama_settings (model, base_url)
        self.llm = OllamaLLM (model = model, base_url = base_url)

    def entity_extraction (self, hypothetical_answer_list, do_graph_search):
        
        if not do_graph_search:
            return []
        
        entities_list = []
        for user_query in hypothetical_answer_list:

            prompt = f""" System: You are an expert Medical Entity Extractor.
                
                Task: Identify entities from the following categories:
                - DISEASE: Pathological conditions, illnesses, or symptoms (e.g., Hypertension, Fever).
                - MEDICATION: Drugs, pharmaceuticals, or treatments (e.g., Insulin, Aspirin).

                Rules:
                1. Only extract entities mentioned in the query.
                2. Normalize terms to their standard medical names if possible.
                3. If no entities are found, return an empty list.

                Output Format Example:
                Query: "Is Ibuprofen okay for my migraine?"
                Output: DISEASE: ["Migraine"] & MEDICATION: ["Ibuprofen"]

                Query: "{user_query}"
                Output: (YOUR RESPONSE)
            """
            
            response = self.llm.invoke (prompt)

            sections = {"disease": [], "medication": []}
            for match in re.finditer (r'\b(DISEASE|MEDICATION)\s*:\s*\[([^\]]*)\]', response, re.IGNORECASE):
                category = match.group (1).lower ()
                names = re.findall (r'"([^"]+)"', match.group (2))
                sections[category].extend (name.lower ().strip () for name in names if name.strip ())

            disease_entities = list (dict.fromkeys (sections["disease"]))
            medication_entities = list (dict.fromkeys (sections["medication"]))

            entities_list.append ([disease_entities, medication_entities])
        
        return entities_list

    def user_query_understanding (self, user_query, chat_history, do_expansion, do_rewrite, do_HyDE, replace_query_by_HyDE):

        intent = self.intent_detection (user_query, chat_history)

        if do_rewrite:
            rewrite_query = self.rewrite (user_query, chat_history)
        else:
            rewrite_query = user_query

        if intent == "RAG_SEARCH":
            if do_expansion:
                expand_query = self.expansion (rewrite_query, chat_history)
            else:
                expand_query = [rewrite_query]
            
            if do_HyDE:
                hypothetical_answer_list = self.HyDE (expand_query, chat_history, len (expand_query))
            else:
                hypothetical_answer_list = expand_query
            
            if replace_query_by_HyDE:
                return hypothetical_answer_list, hypothetical_answer_list[0], intent

            return hypothetical_answer_list, rewrite_query, intent

        else:
            return None, rewrite_query, intent
    
    def intent_detection (self, user_query, chat_history):
        
        prompt = f""" SYSTEM: You are an Intent Detection Agent for a RAG Query Processor. Analyze the user's input query and chat history to classify it into one of these exact categories: [RAG_SEARCH, CHITCHAT]. Base your decision on whether the query seeks document retrieval (RAG_SEARCH), casual conversation (CHITCHAT). Respond concisely with only the category.
            
            USER INPUT: {user_query}
            CHAT HISTORY: {chat_history}

            Return your response in the following format:
            CATEGORY: (TYPE)
        """

        response = self.llm.invoke (prompt)
        response = response.strip ().upper ()

        if "CHITCHAT" in response:
            return "CHITCHAT"
        
        return "RAG_SEARCH"

    def rewrite (self, user_query, chat_history):

        prompt = f"""SYSTEM: You are a Query Rewrite Agent for a RAG Query Processor. Take the user's follow-up question and preceding conversation history to create a single, standalone search query that encapsulates all necessary context from the chat. Ensure the rewrite is self-contained, concise, and optimized for retrieving relevant documents from a vector database. Ignore chit-chat or non-search intents.
    
            USER INPUT: {user_query}
            CHAT HISTORY: {chat_history}

            Return your response in the following format:
            REWRITE: 
        """
        
        response = self.llm.invoke (prompt)

        if "REWRITE:" in response:
            response = response.split ("REWRITE:")[-1].strip ()
            response = re.split (r'\n', response)[0].strip ()
        else:
            return response.strip()

        return response

    def expansion (self, rewritten_query, chat_history):

        prompt = f""" SYSTEM: You are a Query Expansion Agent for a RAG Query Processor. Given a standalone rewritten query, generate exactly 1 variation of it. Each variation should be a slight rephrasing or expansion to capture synonyms, related terms, or alternative framings that maximize retrieval of relevant documents from a vector database. Keep them concise and focused on the core information need.
            
            USER INPUT: {rewritten_query}
            CHAT HISTORY: {chat_history}

            Return your response in the following format:
            EXPANSIONS: 
            1. (Variation 1)
        """
        
        response = self.llm.invoke (prompt)

        if "1." in response:
            response = response.split ("1.")
        elif "EXPANSIONS:" in response:
            response = response.split ("EXPANSIONS:")
        else:
            return [response.strip (), rewritten_query]
        
        return [response[-1].strip (), rewritten_query]

    def HyDE (self, expansion_rewrite, chat_history, hyde):

        prompt = f""" SYSTEM: You are a Hypothetical Answer Agent for a RAG Query Processor. Generate a concise, expert hypothetical answer for 1 query provided.
            
            CHAT HISTORY: {chat_history}
            EXPANSIONS: {expansion_rewrite[0]}

            Output Format: (1-3 sentences)
        """
        
        response = self.llm.invoke (prompt)
        hypothetical_answer_list = [response.strip ()]

        while len (hypothetical_answer_list) < hyde:
            hypothetical_answer_list.append (expansion_rewrite[-1])

        return hypothetical_answer_list[ : hyde]

    def extractive_compression (self, rewritten_query, chunk_content_list):

        summaries = []
        for chunk_content in chunk_content_list:

            prompt = f""" SYSTEM: You are a Medical Data Analyst. Your task is to summarize the provided document chunk specifically in the context of the user's search query. 
                Extract only the clinical facts, diagnostic criteria, or treatment protocols mentioned. 
                Maintain professional medical terminology. If the chunk is not relevant to the query, respond with "NOT_RELEVANT".

                REWRITTEN QUERY: {rewritten_query}
                DOCUMENT CHUNK: {chunk_content}

                SUMMARY: (Provide a concise, 2-5 sentence summary of the relevant medical facts)
            """
            response = self.llm.invoke (prompt)

            response = response.strip ()
            if response.upper ().startswith ("SUMMARY:"):
                response = response[len ("SUMMARY:"):].strip ()
            if not response or response.upper ().rstrip (" .") == "NOT_RELEVANT":
                continue
            summaries.append (f"- {response}{'' if response.endswith (('.', '?', '!')) else '.'}")

        return "\n".join (summaries)

    def ordering (self, chunks, do_ordering, top_k_chunk = 5):

        priority_chunks = ""
        subordinate_chunks = ""
        
        if do_ordering and top_k_chunk <= len (chunks):
            for idx1 in range (top_k_chunk):
                chunk_content = chunks[idx1].page_content
                priority_chunks += f"- {chunk_content}.\n"

            for idx2 in range (top_k_chunk, len (chunks)):
                chunk_content = chunks[idx2].page_content
                subordinate_chunks += f"- {chunk_content}.\n"

        else:
            for idx in range (min (len (chunks), top_k_chunk)):
                chunk_content = chunks[idx].page_content
                priority_chunks += f"- {chunk_content}.\n"    
        
        if subordinate_chunks == "":
            return [priority_chunks]
        
        return [priority_chunks, subordinate_chunks]

    def context_retrieval_processing (self, chunks, rewritten_query, do_ordering, do_extractive_compression): 

        chunk_content_ordered = self.ordering (chunks, do_ordering)

        if do_extractive_compression:
            chunk_content_summarized = self.extractive_compression (
                rewritten_query, [chunk.page_content for chunk in chunks[:5]]
            )
        elif len (chunk_content_ordered) == 2:
            chunk_content_summarized = chunk_content_ordered[0] + chunk_content_ordered[1]
        else:
            chunk_content_summarized = chunk_content_ordered[0]

        return chunk_content_summarized
