from Augmented_Generation import RAG
from Retrieval import Retriever
import streamlit as st
import logging


st.set_page_config (page_title = "Healthcare Chatbot", page_icon = "🤖")

@st.cache_resource
def load_retriever ():
    return Retriever ()

if "rag" not in st.session_state:
    placeholder = st.empty ()

    with placeholder.container ():
        st.info ("Loading retrieval models. The first start can take about a minute.")
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

def show_search_warnings (warnings):
    for warning in warnings:
        st.warning (warning)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message (message["role"]):
        st.markdown (message["content"])
        show_search_warnings (message.get ("search_warnings", []))
        show_graph_warning (message.get ("graph_warning", ""))
        show_retrieved_context (message.get ("context", []))

if prompt := st.chat_input ("Ask me anything..."):
    st.chat_message ("user").markdown (prompt)
    st.session_state.messages.append ({"role": "user", "content": prompt})

    with st.chat_message ("assistant"):
        with st.status ("Preparing response...", expanded = False) as status:
            try:
                response = rag.RAG_Online_Phase (prompt)
                status.update (label = "Response ready", state = "complete")
            except Exception as error:
                logging.getLogger (__name__).warning ("Request failed: %s", type (error).__name__)
                status.update (label = "Request failed", state = "error")
                st.error ("The request could not be completed. Please try again.")
                response = None

        if response:
            st.markdown (response)
            show_search_warnings (rag.search_warnings)
            show_graph_warning (rag.graph_warning)
            context = rag.response_context_snapshot ()
            show_retrieved_context (context)
            st.session_state.messages.append ({
                "role": "assistant", "content": response,
                "context": context, "graph_warning": rag.graph_warning,
                "search_warnings": rag.search_warnings.copy (),
            })
            try:
                rag.RAG_PostOnline_Phase ()
            except Exception:
                st.warning ("The answer was shown, but conversation follow-up processing failed.")

with st.sidebar:
    startup_warnings = getattr (rag.retriever, "startup_warnings", [])
    if startup_warnings:
        st.header ("Retrieval status")
        for warning in startup_warnings:
            st.warning (warning)

    st.header ("Settings")

    if st.button ("Clear This Session"):
        rag.clear_conversation ()
        st.session_state.messages = []
        st.rerun ()

    if rag.log_interactions:
        st.caption ("Clearing this session does not erase the optional Chat_History.log file.")
