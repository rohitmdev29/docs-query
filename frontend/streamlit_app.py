import streamlit as st
import requests

# Backend API URL
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Docs-Query", page_icon="📚", layout="wide")

# Thoda custom CSS for better spacing
st.markdown("""
    <style>
    .block-container { padding-bottom: 5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Docs-Query")
st.caption("Upload a PDF, ask questions, and watch the RAG pipeline work.")

# Setup session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to get the list of documents from the backend
def fetch_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            st.session_state.documents = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend is down. Is FastAPI running on port 8000?")

# Load documents on start
if not st.session_state.documents:
    fetch_documents()


# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    
    if st.button("Process PDF", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Embedding document into vector database..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Success! Created {data['chunks']} chunks.")
                        st.session_state.active_doc_id = data["doc_id"]
                        fetch_documents()
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please select a file first.")

    st.divider()

    st.header("2. Select Document")
    
    if st.session_state.documents:
        # Map filenames to their IDs
        doc_options = {doc["source"]: doc["doc_id"] for doc in st.session_state.documents}
        
        # Find which one should be selected by default
        default_index = 0
        if st.session_state.active_doc_id:
            for i, (name, d_id) in enumerate(doc_options.items()):
                if d_id == st.session_state.active_doc_id:
                    default_index = i
                    break
                    
        selected_name = st.selectbox("Active document:", list(doc_options.keys()), index=default_index)
        st.session_state.active_doc_id = doc_options[selected_name]
    else:
        st.info("No documents in database yet.")

    st.divider()

    # NEW: Search type option add kiya hai
    st.header("3. Search Settings")
    selected_search = st.radio(
        "Search Type:",
        options=["hybrid", "semantic"],
        format_func=lambda x: "Hybrid (BM25 + Vector)" if x == "hybrid" else "Semantic (Vector only)",
        help="Hybrid combines keyword and meaning search for better accuracy."
    )

    # Button to clear chat history
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --- MAIN CHAT AREA ---
st.subheader("Ask a Question")

# Show past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**[{src['index']}]** `{src['source']}` — Page {src['page']}")
                    st.caption(f"> {src['text']}")

# Chat input
if prompt := st.chat_input("E.g., What is Rohit's experience with LLMs?"):
    if not st.session_state.active_doc_id:
        st.warning("Pehle document upload aur select karo!")
    else:
        # 1. Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call Backend & Show Pipeline
        with st.chat_message("assistant"):
            # Yahan hum RAG pipeline ka flow dikha rahe hain
            with st.status("Running RAG Pipeline...", expanded=True) as status:
                
                # Status text ko search type ke hisaab se change kiya
                if selected_search == "hybrid":
                    st.write("🔍 1. Running Hybrid Search (BM25 + Vector)...")
                else:
                    st.write("🔍 1. Running Semantic Search (Vector only)...")
                
                payload = {
                    "question": prompt,
                    "doc_id": st.session_state.active_doc_id,
                    "top_k": 3,
                    "search_type": selected_search  # Backend ko bhej rahe hain
                }
                
                try:
                    st.write("📂 2. Retrieving relevant chunks...")
                    
                    # API Call
                    res = requests.post(f"{API_URL}/query", json=payload)
                    
                    if res.status_code == 200:
                        st.write("🧠 3. Generating answer with LLM...")
                        status.update(label="Pipeline Complete!", state="complete", expanded=False)
                        
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        # Show Answer
                        st.markdown(answer)
                        
                        # Show Sources
                        with st.expander("View Sources"):
                            for src in sources:
                                st.markdown(f"**[{src['index']}]** `{src['source']}` — Page {src['page']}")
                                st.caption(f"> {src['text']}")
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        status.update(label="Error", state="error", expanded=False)
                        st.error(f"Error from server: {res.text}")
                        
                except Exception as e:
                    status.update(label="Error", state="error", expanded=False)
                    st.error(f"Failed to connect to API: {e}")