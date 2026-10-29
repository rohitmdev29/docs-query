import streamlit as st
import requests

# API base URL. Make sure uvicorn is running on port 8000.
API_URL = "http://localhost:8000/api/v1"

# Set up the page layout
st.set_page_config(page_title="Docs-Query", page_icon="📚", layout="wide")

# Custom CSS to make chat input stick to the bottom a bit better
st.markdown("""
    <style>
    .block-container { padding-bottom: 5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Docs-Query")
st.caption("Upload a PDF, ask questions, and get answers with exact citations.")

# Initialize session state variables so they don't reset on every click
if "documents" not in st.session_state:
    st.session_state.documents = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Helper function to fetch the list of uploaded documents
def fetch_documents():
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            st.session_state.documents = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Is FastAPI running?")


# Fetch documents on first load
if not st.session_state.documents:
    fetch_documents()


# --- SIDEBAR: File Upload & Document Selection ---
with st.sidebar:
    st.header("1. Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if st.button("Process PDF", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing and embedding document..."):
                # Wrap file in a tuple to send via requests
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Done! Added {data['chunks']} chunks.")
                        st.session_state.active_doc_id = data["doc_id"]
                        fetch_documents() # Refresh the list
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Network error: {e}")
        else:
            st.warning("Please choose a file first.")

    st.divider()

    st.header("2. Select Document")
    
    if st.session_state.documents:
        # Create a mapping of filename to doc_id for the dropdown
        doc_options = {doc["source"]: doc["doc_id"] for doc in st.session_state.documents}
        
        # Figure out the current index. If we just uploaded, select that one.
        default_index = 0
        if st.session_state.active_doc_id:
            for i, (name, d_id) in enumerate(doc_options.items()):
                if d_id == st.session_state.active_doc_id:
                    default_index = i
                    break
                    
        selected_name = st.selectbox("Active document:", list(doc_options.keys()), index=default_index)
        st.session_state.active_doc_id = doc_options[selected_name]
    else:
        st.info("No documents uploaded yet.")


# --- MAIN AREA: Chat Interface ---
st.subheader("Ask a Question")

# Display previous chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the assistant has sources, show them in an expander
        if msg.get("sources"):
            with st.expander("View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**[{src['index']}]** `{src['source']}` — Page {src['page']}")
                    st.caption(f"> {src['text']}")

# The chat input box at the bottom
if prompt := st.chat_input("E.g., What is Rohit's experience with LLMs?"):
    if not st.session_state.active_doc_id:
        st.warning("Please upload and select a document first.")
    else:
        # 1. Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the backend API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "question": prompt,
                    "doc_id": st.session_state.active_doc_id,
                    "top_k": 3
                }
                
                try:
                    res = requests.post(f"{API_URL}/query", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        st.markdown(answer)
                        
                        # Show sources in an expander
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
                        st.error(f"Error from server: {res.text}")
                        
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")