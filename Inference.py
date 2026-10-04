from Hybrid_Dual_Indexing import Keyword_Search, Semantic_Search
from Augmented_Generation import RAG
from Retrieval import Retriever
import streamlit as st
import time


st.set_page_config (page_title = "Healthcare Chatbot", page_icon = "🤖")

@st.cache_resource
def load_retriever ():
    return Retriever ()

if "rag" not in st.session_state:
    placeholder = st.empty ()

    with placeholder.container ():
        st.warning ("⚠️ **THIS MEDICAL RAG SYSTEM SHOULD TAKE 60s TO LOAD ! APOLOGIZE FOR THE SLOW.**")
        rag = RAG (retriever = load_retriever ())
        
    placeholder.empty ()
    st.session_state.rag = rag

rag = st.session_state.rag

st.title ("🤖 Healthcare AI Assistant with RAG")
st.warning (
    "Educational demo only. This app cannot diagnose conditions or guide treatment. "
    "For a medical emergency, contact local emergency services."
)

def show_retrieved_context (context):
    if not context:
        return
    with st.expander ("Retrieved context used for this answer"):
        st.caption ("These are retrieved dataset records, not citations for individual claims.")
        for index, item in enumerate (context, start = 1):
            label = f"{index}. {item['kind']} — {item['source']}"
            if item["disease"]:
                label += f" ({item['disease']})"
            st.write (label)
            st.text (item["text"])

def show_graph_warning (warning):
    if warning:
        st.warning (warning)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message (message["role"]):
        st.markdown (message["content"])
        show_graph_warning (message.get ("graph_warning", ""))
        show_retrieved_context (message.get ("context", []))

if prompt := st.chat_input ("Ask me anything..."):
    st.chat_message ("user").markdown (prompt)
    st.session_state.messages.append ({"role": "user", "content": prompt})

    with st.chat_message ("assistant"):
        with st.status ("Analyzing...", expanded = False) as status:
            time.sleep (1.0)
            
            status.update (label = "Retrieving...", state = "running")
            time.sleep (1.0) 
            
            status.update (label = "Feedback loop...", state = "running")
            
            try:
                response = rag.RAG_Online_Phase (prompt)
                status.update (label = "Analysis Complete", state = "complete")
                
            except Exception as e:
                status.update (label = "Error Occurred", state = "error")
                st.error (f"An error occurred: {e}")
                response = None

        if response:
            st.markdown (response)
            show_graph_warning (rag.graph_warning)
            context = rag.response_context_snapshot ()
            show_retrieved_context (context)
            st.session_state.messages.append ({
                "role": "assistant", "content": response,
                "context": context, "graph_warning": rag.graph_warning,
            })
            try:
                rag.RAG_PostOnline_Phase ()
            except Exception:
                st.warning ("The answer was shown, but conversation follow-up processing failed.")

with st.sidebar:
    st.header ("Settings")

    if st.button ("Clear Chat History"):
        if hasattr (rag, 'chat_history'):
            rag.chat_history = "No prior conversation"

        st.session_state.messages = []
        st.rerun ()
