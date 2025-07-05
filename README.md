# Multi-User Document Search & Conversational Q&A System (RAG)

A Streamlit web app for multi-user, access-controlled document Q&A using **completely local models** (no API costs or rate limits).

## Features
- Simulated user login (by email)
- User-specific document access (from `user_access.yaml`)
- PDF ingestion and chunking
- RAG: retrieves relevant document chunks using embeddings
- **Local embeddings** using sentence-transformers (no API costs)
- **Local LLM** using Ollama (no API costs)
- Conversational Q&A with context maintenance
- **Caching system** for chunks and embeddings (faster startup, cost savings)
- Simple, clean UI

## Models Used

### 1. **Embedding Models**

#### **Primary: all-MiniLM-L6-v2 (Sentence Transformers)**
- **Purpose**: Convert text chunks and queries into numerical vectors for similarity search
- **Why this model**: 
  - **Speed**: 384-dimensional embeddings (vs OpenAI's 1536) = faster processing
  - **Accuracy**: Excellent performance on semantic similarity tasks
  - **Cost**: Free, runs locally - no API costs or rate limits
  - **Size**: ~90MB download, reasonable for local deployment
- **Alternative considered**: OpenAI's text-embedding-ada-002 (higher quality but expensive)

#### **Fallback: OpenAI text-embedding-ada-002**
- **Purpose**: Backup embedding model when local model fails
- **Why included**: Ensures system reliability even if local model has issues
- **Usage**: Only used when `USE_LOCAL_EMBEDDINGS = False` in `utils.py`

### 2. **Large Language Model (LLM)**

#### **Primary: Llama 2 (via Ollama)**
- **Purpose**: Generate human-like answers based on retrieved document chunks
- **Why this model**:
  - **Cost**: Completely free, runs locally
  - **Quality**: Good performance for Q&A tasks
  - **Reliability**: No API rate limits or downtime
  - **Privacy**: All data stays on your machine
- **Alternative**: OpenAI GPT-3.5-turbo (higher quality but expensive)

### 3. **PDF Processing**

#### **PyPDF2**
- **Purpose**: Extract text content from PDF documents
- **Why this library**:
  - **Simplicity**: Easy to use and integrate
  - **Reliability**: Well-maintained and stable
  - **Performance**: Fast text extraction for most PDF formats
- **Alternative considered**: pdfplumber (better for complex layouts but heavier)

### 4. **Text Chunking Strategy**

#### **Sliding Window with Overlap**
- **Chunk size**: 2000 characters (~500-700 words)
- **Overlap**: 200 characters
- **Why this approach**:
  - **Context preservation**: Overlap ensures important information isn't split
  - **Optimal size**: Balances embedding efficiency with context completeness
  - **RAG-friendly**: Chunks are sized appropriately for LLM context windows

## Setup Instructions

### **Step 1: Install Python Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Install Ollama (Local LLM)**

#### **Windows:**
1. **Download Ollama:**
   - Go to [ollama.ai](https://ollama.ai)
   - Click "Download for Windows"
   - Download the `.exe` installer

2. **Install Ollama:**
   - Run the downloaded `.exe` file
   - Follow the installation wizard
   - Ollama will be installed as a Windows service

3. **Verify Installation:**
   ```bash
   ollama --version
   ```

#### **Mac/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### **Step 3: Download LLM Models**

After installing Ollama, download the required models:

```bash
# Download Llama 2 (recommended, ~4GB)
ollama pull llama2

# Alternative models (optional):
ollama pull mistral    # Smaller, faster (~2GB)
ollama pull codellama  # Good for technical content (~4GB)
ollama pull phi        # Very small, fast (~1.5GB)
```

**Note:** First download may take 5-15 minutes depending on your internet speed.

### **Step 4: Start Ollama Service**

#### **Windows:**
- Ollama should start automatically as a Windows service
- If not, open Command Prompt as Administrator and run:
  ```bash
  ollama serve
  ```

#### **Mac/Linux:**
```bash
ollama serve
```

**Keep this running in the background while using the app.**

### **Step 5: Test Ollama Connection**
```bash
# Test if Ollama is working
ollama list
```

You should see your downloaded models listed.

### **Step 6: Add Sample PDFs**
- Create a `data/` folder in your project directory
- Place 5+ earnings call PDFs in the `data/` folder
- Name them as `CompanyA.pdf`, `CompanyB.pdf`, etc.

### **Step 7: Edit User Access**
- Update `user_access.yaml` to control which users can access which PDFs
- Example:
  ```yaml
  alice@email.com:
    - CompanyA.pdf
  bob@email.com:
    - CompanyB.pdf
    - CompanyC.pdf
  ```

### **Step 8: (Optional) Use OpenAI API Fallback**
- If you want to use OpenAI for embeddings or LLM, set `USE_LOCAL_EMBEDDINGS = False` or `USE_LOCAL_LLM = False` in `utils.py`.
- Add your OpenAI API key to a `.env` file:
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```

### **Step 9: Run the App**
```bash
streamlit run app.py
```

## Demo Users
- `alice@email.com` → CompanyA.pdf
- `bob@email.com` → CompanyB.pdf, CompanyC.pdf
- `charlie@email.com` → CompanyD.pdf, CompanyE.pdf

## How RAG Works
- All PDFs are chunked and embedded using sentence-transformers (local).
- When a user asks a question, the system retrieves the most relevant chunks (by cosine similarity) from their allowed documents.
- Only those chunks (plus chat context) are sent to the local LLM for answer generation.
- This ensures efficient, accurate, and access-controlled Q&A with zero API costs.

## Model Options
- **Local Embeddings (Default)**: Uses `all-MiniLM-L6-v2` from sentence-transformers
  - ✅ No API costs
  - ✅ Runs locally
  - ✅ Fast and accurate
- **Local LLM (Default)**: Uses `llama2` via Ollama
  - ✅ No API costs
  - ✅ Runs locally
  - ✅ Good quality responses
- **OpenAI Fallback**: Available by setting `USE_LOCAL_LLM = False` in `utils.py`
  - ⚠️ Requires API key and credits
  - ✅ Higher quality responses

## Caching System
- **Chunks** are cached in `cached_chunks.pkl`
- **Embeddings** are cached in `cached_embeddings.pkl`
- **Benefits**:
  - Faster app startup after first run
  - Reduced processing time (embeddings computed once)
  - Better user experience
- **Cache Management**:
  - Use "Refresh Cache" button in sidebar to clear cache
  - Restart app after adding new PDFs to reprocess

## Requirements
- streamlit>=1.28.0
- openai>=1.0.0 (optional, for fallback)
- PyPDF2>=3.0.0
- python-dotenv>=1.0.0
- pyyaml>=6.0
- numpy>=1.24.0
- sentence-transformers>=2.2.0
- torch>=2.0.0
- transformers>=4.30.0
- ollama>=0.1.0

## Usage
- Login with a sample email
- Ask questions about your accessible documents
- Follow up with context-aware queries
- Each user only sees their own documents and chat

## Troubleshooting

### **Ollama Issues:**
- **"Failed to connect to Ollama"**: 
  - Ensure Ollama is installed and running
  - On Windows: Check if Ollama service is running in Task Manager
  - Run `ollama serve` in a separate terminal
- **"Model not found"**: 
  - Run `ollama pull llama2` to download the model
  - Check available models with `ollama list`
- **"Permission denied"**: 
  - Run Command Prompt as Administrator on Windows
  - Ensure you have write permissions to the Ollama directory

### **Model Download Issues:**
- **Slow download**: Models are large (1-4GB), be patient
- **Download fails**: Check internet connection and try again
- **Insufficient disk space**: Ensure you have at least 10GB free space

### **App Issues:**
- **Cache Issues**: Use "Refresh Cache" button to clear and rebuild cache
- **New PDFs**: Clear cache and restart app to include new documents
- **Local Model Download**: First run will download the sentence-transformers model (~90MB)

### **Performance Tips:**
- **First run**: Will be slow as models download and cache builds
- **Subsequent runs**: Much faster due to caching
- **Memory usage**: LLM models use 2-8GB RAM depending on model size
- **GPU acceleration**: Ollama automatically uses GPU if available

## Alternative Models

If you want to try different models, you can change `LOCAL_LLM_MODEL` in `utils.py`:

```python
LOCAL_LLM_MODEL = "mistral"    # Smaller, faster
LOCAL_LLM_MODEL = "codellama"  # Good for technical content
LOCAL_LLM_MODEL = "phi"        # Very small, fast
```

Then download the model:
```bash
ollama pull mistral
```

---

**Note:**
- This is a demo. For production, add real authentication and security.
- **Zero API costs**: All models run locally.
- **No rate limits**: No external API dependencies.
- **Privacy**: All data stays on your machine.
- **System requirements**: At least 8GB RAM and 10GB free disk space recommended. 