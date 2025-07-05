import streamlit as st
import os
import time
from utils import (
    load_pdfs, chunk_text, get_user_docs, get_pdf_text,
    load_user_access, embed_chunks, retrieve_top_k,
    save_chunks, load_chunks, save_embeddings, load_embeddings,
    USE_LOCAL_EMBEDDINGS, LOCAL_MODEL_NAME, USE_LOCAL_LLM, LOCAL_LLM_MODEL,
    get_llm_response
)
import openai
import yaml
import numpy as np
from dotenv import load_dotenv
import re

# --- Config ---
DATA_DIR = "data"
USER_ACCESS_PATH = "user_access.yaml"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 2
EMBED_MODEL = "text-embedding-ada-002"

# --- Load env vars ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load user access from YAML ---
@st.cache_resource(show_spinner=True)
def get_user_access():
    return load_user_access(USER_ACCESS_PATH)

user_access = get_user_access()

# --- Load and chunk all PDFs (with caching) ---
@st.cache_resource(show_spinner=True)
def get_all_chunks_and_texts():
    # Try to load from cache first
    cached_chunks = load_chunks()
    if cached_chunks is not None:
        st.info("📁 Loaded chunks from cache")
        return cached_chunks, cached_chunks  # Return chunks for both text and chunks
    
    # If no cache, process PDFs
    st.info("🔄 Processing PDFs and creating chunks...")
    pdf_texts = load_pdfs(DATA_DIR)
    doc_chunks = {}
    for fname, text in pdf_texts.items():
        doc_chunks[fname] = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Save to cache
    save_chunks(doc_chunks)
    st.success("💾 Chunks saved to cache")
    return pdf_texts, doc_chunks

pdf_texts, doc_chunks = get_all_chunks_and_texts()

# --- Embed all chunks for all docs (with caching) ---
@st.cache_resource(show_spinner=False)
def get_all_embeddings(doc_chunks):
    # Try to load from cache first
    cached_embeddings = load_embeddings()
    if cached_embeddings is not None:
        st.info("📁 Loaded embeddings from cache")
        return cached_embeddings
    
    # If no cache, compute embeddings
    if USE_LOCAL_EMBEDDINGS:
        st.info(f"🔄 Computing embeddings using {LOCAL_MODEL_NAME}...")
    else:
        st.info("🔄 Computing embeddings using OpenAI...")
    
    doc_embeddings = {}
    for fname, chunks in doc_chunks.items():
        if chunks:
            doc_embeddings[fname] = embed_chunks(chunks, model=EMBED_MODEL)
        else:
            # Set correct dimension based on embedding model
            if USE_LOCAL_EMBEDDINGS:
                dim = 384  # all-MiniLM-L6-v2 dimension
            else:
                dim = 1536  # OpenAI ada-002 dimension
            doc_embeddings[fname] = np.zeros((0, dim))
    
    # Save to cache
    save_embeddings(doc_embeddings)
    st.success("💾 Embeddings saved to cache")
    return doc_embeddings

doc_embeddings = get_all_embeddings(doc_chunks)

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-User Document Q&A (RAG)", layout="wide")
st.title("📄 Multi-User Document Search & Conversational Q&A (RAG)")

# --- Cache management in sidebar ---
st.sidebar.markdown("### Cache Management")
if st.sidebar.button("🔄 Refresh Cache"):
    # Clear cache files
    for cache_file in ["cached_chunks.pkl", "cached_embeddings.pkl"]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
    st.sidebar.success("Cache cleared! Restart the app to reprocess PDFs.")
    st.rerun()

# --- Model info in sidebar ---
st.sidebar.markdown("### Models Used")
if USE_LOCAL_EMBEDDINGS:
    st.sidebar.success(f"✅ Embeddings: {LOCAL_MODEL_NAME}")
    st.sidebar.info("No API costs, runs locally")
else:
    st.sidebar.warning("⚠️ Embeddings: OpenAI API")
    st.sidebar.info("Requires API key and credits")

if USE_LOCAL_LLM:
    st.sidebar.success(f"✅ LLM: {LOCAL_LLM_MODEL}")
    st.sidebar.info("No API costs, runs locally")
    
    # Performance tips
    st.sidebar.markdown("### ⚡ Performance Tips")
    st.sidebar.info("""
    **For faster responses:**
    - Use `llama2:7b-chat-q4_K_M` (faster, quantized)(current)
    - Use `llama2:7b` (good balance)
    - Use `mistral` (slower but better quality)
    """)
    
else:
    st.sidebar.warning("⚠️ LLM: OpenAI API")
    st.sidebar.info("Requires API key and credits")

# --- Login ---
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.chat_history = []

if st.session_state.user is None:
    st.subheader("Login")
    email = st.text_input("Enter your email:")
    if st.button("Login"):
        if email in user_access:
            st.session_state.user = email
            st.session_state.chat_history = []
            st.success(f"Logged in as {email}")
            st.rerun()
        else:
            st.error("Unauthorized email. Try one of the sample users.")
    st.info("Sample users: alice@email.com, bob@email.com, charlie@email.com")
    st.stop()

user = st.session_state.user
user_docs = get_user_docs(user, user_access)

st.sidebar.write(f"**Logged in as:** {user}")
st.sidebar.write(f"**Accessible documents:**")
for doc in user_docs:
    st.sidebar.write(f"- {doc}")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.chat_history = []
    st.rerun()

# --- Q&A Section ---
st.subheader("Ask a question about your documents")
query = st.text_input("Your question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_company_names_from_query(query, all_company_names):
    # Simple case-insensitive match for any company name in the query
    found = set()
    query_lower = query.lower()
    for name in all_company_names:
        if name.lower() in query_lower:
            found.add(name)
    return found

if st.button("Submit") and query:
    # Performance monitoring
    start_time = time.time()
    
    # --- Company name access control check ---
    # Get all possible company names (from all PDFs)
    all_company_names = [os.path.splitext(f)[0] for f in doc_chunks.keys()]
    user_company_names = [os.path.splitext(f)[0] for f in user_docs]
    mentioned_companies = extract_company_names_from_query(query, all_company_names)
    unauthorized = [c for c in mentioned_companies if c not in user_company_names]
    if unauthorized:
        st.markdown("### Response:")
        st.error(f"Access Denied: You do not have permission to view information about: {', '.join(unauthorized)}")
        st.session_state.chat_history.append((query, f"Access Denied: You do not have permission to view information about: {', '.join(unauthorized)}"))
        st.rerun()
    
    # Step 1: Gather chunks and embeddings
    with st.spinner("🔍 Gathering document chunks..."):
        all_chunks = []
        all_embeddings = []
        for doc in user_docs:
            all_chunks.extend(doc_chunks.get(doc, []))
            if doc in doc_embeddings and len(doc_embeddings[doc]) > 0:
                all_embeddings.append(doc_embeddings[doc])
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
        else:
            # Set correct dimension based on embedding model
            if USE_LOCAL_EMBEDDINGS:
                dim = 384  # all-MiniLM-L6-v2 dimension
            else:
                dim = 1536  # OpenAI ada-002 dimension
            all_embeddings = np.zeros((0, dim))
    
    chunk_time = time.time() - start_time
    st.info(f"📊 Found {len(all_chunks)} chunks in {chunk_time:.1f}s")
    
    # Step 2: Retrieve relevant chunks
    with st.spinner("🎯 Finding relevant document sections..."):
        if len(all_chunks) > 0 and all_embeddings.shape[0] > 0:
            top_chunks = retrieve_top_k(query, all_chunks, all_embeddings, k=TOP_K, model=EMBED_MODEL)
            context = "\n---\n".join(top_chunks)
        else:
            top_chunks = []
            context = "[No accessible document content found.]"
    
    retrieval_time = time.time() - start_time - chunk_time
    st.info(f"📊 Retrieved {len(top_chunks) if 'top_chunks' in locals() else 0} relevant sections in {retrieval_time:.1f}s")
    
    # Step 3: Build prompt
    chat_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-3:]])
    if not top_chunks or context == "[No accessible document content found.]":
        prompt = (
            "You are a helpful assistant. "
            "If the following document excerpts do not contain information about the user's question, "
            "or if the user asks about a company they do not have access to, reply with: "
            "'Access Denied: You do not have permission to view information about this company.'\n\n"
            f"{context}\n\n{chat_context}\nQ: {query}\nA:"
        )
    else:
        prompt = (
            "You are a helpful assistant. Use ONLY the following document excerpts to answer.\n\n"
            f"{context}\n\n{chat_context}\nQ: {query}\nA:"
        )
    
    # Step 4: Get response from LLM
    with st.spinner("🤖 Generating AI response (this may take some time based on the Model used)..."):
        answer = get_llm_response(prompt)
    
    # Display the response immediately
    st.markdown("### Response:")
    if len(answer) > 500:
        st.text_area("Full Response:", value=answer, height=300, key="current_response")
    else:
        st.markdown(f"**{answer}**")
    
    # Performance summary
    total_time = time.time() - start_time
    st.success(f"✅ Response generated in {total_time:.1f} seconds")
    
    st.session_state.chat_history.append((query, answer))
    st.rerun()

# --- Chat History ---
st.markdown("---")
st.subheader("Conversation History")
if st.session_state.chat_history:
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**You:** {q}")
        # Use text_area for longer responses to ensure full display
        if len(a) > 500:
            st.text_area(f"**Assistant Response:**", value=a, height=200, key=f"response_{i}")
        else:
            st.markdown(f"**Assistant:** {a}")
        st.markdown("---")
else:
    st.info("No conversation yet. Ask your first question!")

st.caption("Demo users: alice@email.com, bob@email.com, charlie@email.com. Place sample PDFs in the data/ folder. User access is in user_access.yaml.") 