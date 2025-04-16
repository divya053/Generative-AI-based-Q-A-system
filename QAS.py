import streamlit as st
import os
import tempfile
import requests
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import subprocess

def ensure_model_file():
    model_path = "models/Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf"

    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        st.info("Model not found locally. Downloading from Google Drive...")

        # ✅ Use gdown for Google Drive downloads (make sure to install it)
        file_id = "1v5_eMAlNWBW34ahL2zkH9n8kH0K-UUI1"
        gdown_url = f"https://drive.google.com/uc?id={file_id}"

        try:
            subprocess.run(["gdown", gdown_url, "-O", model_path], check=True)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error("❌ Failed to download model from Google Drive.")
            raise RuntimeError(f"Model download failed: {e}")

    return model_path


class PDF_QA_Model:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4
        )
        self.embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        self.vector_db = None

    def load_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        texts = [doc.page_content for doc in docs if doc.page_content.strip()]
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)

        self.vector_db = FAISS.from_embeddings(list(zip(texts, embeddings)), self.embedding_model)

    def query(self, question: str, max_tokens=200) -> str:
        default_answer = "Sorry, I couldn't find any relevant information in the uploaded document."

        if not self.vector_db:
            return default_answer

        question_embedding = self.embedding_model.encode(question)
        docs = self.vector_db.similarity_search_by_vector(question_embedding, k=3)

        context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
        if not context or len(context.strip()) < 30:
            return default_answer

        prompt = (
            f"Answer the QUESTION using only the CONTEXT below.\n"
            f"If the answer is not in the context, respond: \"{default_answer}\"\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\nANSWER:"
        )

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["\n\n", "QUESTION:"],
            echo=False
        )

        result = output['choices'][0]['text'].strip() if output['choices'] else ""
        return result if result else default_answer


st.set_page_config(page_title="Generative AI-based Q&A System", layout="centered")

st.markdown("<h1 style='text-align: center;'>PDF Q&A System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions based only on the uploaded PDF content.</p>", unsafe_allow_html=True)

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = "upload_1"
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# 🔄 Ensure model file exists before loading model
MODEL_FILE_PATH = ensure_model_file()

@st.cache_resource
def load_model():
    return PDF_QA_Model(MODEL_FILE_PATH)

gpt = load_model()

st.markdown("### Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key=st.session_state.file_uploader_key)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    with st.spinner("Reading and indexing PDF..."):
        gpt.load_pdf(file_path)
        st.session_state.pdf_loaded = True
        st.success("PDF processed successfully!")

# Ask Question
if st.session_state.pdf_loaded:
    st.markdown("---")
    st.markdown("### Ask a Question")
    user_question = st.text_input("Your question:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = gpt.query(user_question)

        st.markdown("### Answer")
        st.markdown(f"<div style='background-color:#333;padding:1rem;border-radius:8px;color:white;'>{answer}</div>", unsafe_allow_html=True)
        st.session_state.history.append(f"Q: {user_question}\nA: {answer}")

# Show History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Q&A History")

    for qa in st.session_state.history:
        st.markdown(f"<div style='background-color:#222;padding:0.75rem;border-radius:8px;color:white;margin-bottom:10px'><pre>{qa}</pre></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download History", data="\n\n".join(st.session_state.history), file_name="qa_history.txt")
    with col2:
        if st.button("Clear All"):
            st.session_state.history = []
            st.session_state.pdf_loaded = False
            st.session_state.file_uploader_key = f"upload_{os.urandom(4).hex()}"
            st.experimental_rerun()
else:
    st.info("Upload a PDF to start asking questions.")
