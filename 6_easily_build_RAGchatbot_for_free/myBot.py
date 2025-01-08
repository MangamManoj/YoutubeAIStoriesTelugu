import os
import streamlit as st
from typing import List
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
import tempfile

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Manoj's Document QA Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
    <style>
        /* General chat container styles */
        .stChat {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .css-1n76uvr {
            width: 100%;
        }
        
        /* Target the message container background */
        .st-emotion-cache-1eqh5qw {
            background-color: #F0F2F6 !important;
        }
        
        /* Target chat message elements specifically */
        div[data-testid="stChatMessage"] {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        /* Style for user messages */
        div[data-testid="stChatMessage"] [data-testid="user"] {
            background-color: #E3F2FD;
        }
        
        /* Style for assistant messages */
        div[data-testid="stChatMessage"] [data-testid="assistant"] {
            background-color: #F5F5F5;
        }
        
        /* Ensure all message content is visible */
        .st-emotion-cache-1eqh5qw p {
            color: #31333F;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your document QA assistant. Please upload your documents in the sidebar to get started."}
        ]
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "index" not in st.session_state:
        st.session_state.index = None

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file temporarily and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def create_or_load_index(file_paths: List[str], rebuild: bool = False) -> VectorStoreIndex:
    """Create or load the vector store index."""
    # Configure LlamaIndex settings
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    storage_dir = "./storage"
    
    if not rebuild and os.path.exists(storage_dir):
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        return load_index_from_storage(storage_context)
    
    # Create new index
    documents = []
    for file_path in file_paths:
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        documents.extend(docs)
    
    # Create index with sentence splitter
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=parser,
    )
    
    # Save index
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    index.storage_context.persist(persist_dir=storage_dir)
    
    return index

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"]
        )
        
        if uploaded_files:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    file_paths.append(file_path)
            
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    st.session_state.index = create_or_load_index(file_paths, rebuild=True)
                    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
                    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                        chat_mode="context",
                        memory=memory,
                        streaming=True
                    )
                st.success("Documents processed successfully!")
    
    # Main chat interface
    st.title("🤖 Manoj's Document QA Assistant")
    st.caption("Ask me anything about your documents or any general questions!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)
        
        if st.session_state.chat_engine is None:
            with st.chat_message("assistant", avatar="🤖"):
                st.write("Please upload documents first to enable the full QA capability!")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload documents first to enable the full QA capability!"})
        else:
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                response_text = ""
                
                # Stream the response
                response = st.session_state.chat_engine.stream_chat(prompt)
                for token in response.response_gen:
                    response_text += token
                    response_placeholder.markdown(response_text + "▌")
                response_placeholder.markdown(response_text)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()