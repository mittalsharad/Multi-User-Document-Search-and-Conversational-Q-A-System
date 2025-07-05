import os
import PyPDF2
import yaml
import openai
import numpy as np
import pickle
import streamlit as st
import time
from dotenv import load_dotenv

# Try to import sentence_transformers, fallback to OpenAI if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import ollama for local LLM
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

load_dotenv()

# --- Cache file paths ---
CHUNKS_CACHE_FILE = "cached_chunks.pkl"
EMBEDDINGS_CACHE_FILE = "cached_embeddings.pkl"

# --- Model selection ---
USE_LOCAL_EMBEDDINGS = True  # Set to False to use OpenAI
USE_LOCAL_LLM = True  # Set to False to use OpenAI
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and accurate
LOCAL_LLM_MODEL = "llama2:7b-chat-q4_K_M"   # "minstral"

# --- YAML loading ---
def load_user_access(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# --- PDF extraction ---
def load_pdfs(data_dir):
    pdf_texts = {}
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    pdf_texts[fname] = text
            except Exception as e:
                pdf_texts[fname] = f"[Error reading PDF: {e}]"
    return pdf_texts

# --- Chunking ---
def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --- Save/Load chunks ---
def save_chunks(chunks_dict, filename=CHUNKS_CACHE_FILE):
    with open(filename, 'wb') as f:
        pickle.dump(chunks_dict, f)

def load_chunks(filename=CHUNKS_CACHE_FILE):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# --- Save/Load embeddings ---
def save_embeddings(embeddings_dict, filename=EMBEDDINGS_CACHE_FILE):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)

def load_embeddings(filename=EMBEDDINGS_CACHE_FILE):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# --- Local embedding model ---
@st.cache_resource(show_spinner=True)
def get_local_embedding_model():
    if SENTENCE_TRANSFORMERS_AVAILABLE and USE_LOCAL_EMBEDDINGS:
        try:
            return SentenceTransformer(LOCAL_MODEL_NAME)
        except Exception as e:
            st.error(f"Error loading local embedding model: {e}")
            return None
    return None

# --- Test Ollama connection ---
def test_ollama_connection():
    if not OLLAMA_AVAILABLE:
        return False, "Ollama Python package not installed"
    
    try:
        # Test with a simple prompt
        response = ollama.chat(
            model=LOCAL_LLM_MODEL, 
            messages=[{'role': 'user', 'content': 'Hello'}],
            options={'timeout': 30}
        )
        return True, "Ollama connection successful"
    except Exception as e:
        return False, f"Ollama connection failed: {str(e)}"

# --- Embedding (Updated for OpenAI v1.0+) ---
def get_embedding(text, model="text-embedding-ada-002"):
    # Always try local embeddings first if enabled
    if USE_LOCAL_EMBEDDINGS:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            local_model = get_local_embedding_model()
            if local_model:
                try:
                    embedding = local_model.encode([text])[0]
                    return np.array(embedding)
                except Exception as e:
                    st.error(f"Error with local embedding: {e}")
                    return None
            else:
                st.error("Local embedding model failed to load")
                return None
        else:
            st.error("sentence-transformers not available. Please install: pip install sentence-transformers")
            return None
    
    # Only use OpenAI if local embeddings are disabled
    if not USE_LOCAL_EMBEDDINGS:
        try:
            resp = openai.embeddings.create(input=[text], model=model)
            return np.array(resp.data[0].embedding)
        except Exception as e:
            st.error(f"OpenAI API error: {e}. Please check your API key and quota.")
            return None
    
    return None

# --- Bulk embedding for all chunks (Optimized) ---
def embed_chunks(chunks, model="text-embedding-ada-002"):
    if USE_LOCAL_EMBEDDINGS and SENTENCE_TRANSFORMERS_AVAILABLE:
        local_model = get_local_embedding_model()
        if local_model:
            try:
                # Batch process all chunks at once (much faster)
                embeddings = local_model.encode(chunks)
                return np.array(embeddings)
            except Exception as e:
                st.error(f"Error with batch embedding: {e}")
                # Fallback to individual processing
                pass
    
    # Fallback to individual processing
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk, model)
        if emb is not None:
            embeddings.append(emb)
        else:
            # If embedding fails, use zero vector
            if USE_LOCAL_EMBEDDINGS and SENTENCE_TRANSFORMERS_AVAILABLE:
                local_model = get_local_embedding_model()
                if local_model:
                    dim = local_model.get_sentence_embedding_dimension()
                else:
                    dim = 384  # all-MiniLM-L6-v2 dimension
            else:
                dim = 1536  # OpenAI ada-002 dimension
            embeddings.append(np.zeros(dim))
    return np.stack(embeddings)

# --- Retrieval: cosine similarity ---
def retrieve_top_k(query, chunk_texts, chunk_embeddings, k=4, model="text-embedding-ada-002"):
    query_emb = get_embedding(query, model)
    if query_emb is None:
        return chunk_texts[:k] if len(chunk_texts) >= k else chunk_texts
    
    sims = np.dot(chunk_embeddings, query_emb) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top_idx = np.argsort(sims)[::-1][:k]
    return [chunk_texts[i] for i in top_idx]

# --- LLM Response Generation ---
def get_llm_response(prompt, model="gpt-3.5-turbo"):
    if USE_LOCAL_LLM and OLLAMA_AVAILABLE:
        # Test connection first
        is_connected, error_msg = test_ollama_connection()
        if not is_connected:
            st.error(f"Ollama connection test failed: {error_msg}")
            return f"[Local LLM error: {error_msg}]"
        
        try:
            # Use timeout and retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = ollama.chat(
                        model=LOCAL_LLM_MODEL, 
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'timeout': 30,  # Reduced timeout for faster response
                            'temperature': 0.1,  # Lower temperature for faster, more focused responses
                            'top_p': 0.8,
                            'num_predict': 200  # Limit response length for faster generation
                        }
                    )
                    return response['message']['content'].strip()
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                        time.sleep(2)  # Wait 2 seconds before retry
                    else:
                        raise e
                        
        except Exception as e:
            error_msg = str(e)
            if "Server disconnected" in error_msg:
                st.error("Ollama server disconnected. Please ensure Ollama is running and try again.")
                return "[Local LLM error: Server disconnected. Please restart Ollama and try again.]"
            elif "timeout" in error_msg.lower():
                st.error("Ollama request timed out. The model may be too slow or overloaded.")
                return "[Local LLM error: Request timed out. Try a shorter question or restart Ollama.]"
            else:
                st.error(f"Ollama error: {error_msg}")
                return f"[Local LLM error: {error_msg}]"
    
    # Fallback to OpenAI
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"

# --- User doc helpers ---
def get_user_docs(user, user_access):
    return user_access.get(user, [])

def get_pdf_text(doc, pdf_texts):
    return pdf_texts.get(doc, "") 