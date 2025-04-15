import streamlit as st
import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import tempfile
import os

class MyGPT4All:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4
        )
        self.embeddings = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.vector_db = None

    def load_documents(self, file_path: str):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(pages)

            texts = [doc.page_content for doc in docs if doc.page_content.strip()]
            embeddings = self.embeddings.encode(texts, batch_size=32, show_progress_bar=False)

            self.vector_db = FAISS.from_embeddings(list(zip(texts, embeddings)), self.embeddings)

    def query(self, question: str, max_tokens=200) -> str:
        default_answer = "I'm sorry, I couldn't find any relevant information in the document."

        if self.vector_db:
            question_embedding = self.embeddings.encode(question)
            docs = self.vector_db.similarity_search_by_vector(question_embedding, k=3)

            if not docs:
                return default_answer

            context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
            if not context:
                return default_answer

            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["Question:", "\n"],
            echo=False
        )

        return output['choices'][0]['text'].strip() if output['choices'] else default_answer


st.set_page_config(page_title="Generative AI-based Q&A system", layout="centered")

st.markdown("<h1 style='text-align: center; color: #ffffff;'>Q&A system</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ffffff;'>Ask questions about your PDF.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_llama_model():
    return MyGPT4All("models/mistral-7b-openorca.Q4_0.gguf")

gpt = load_llama_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

st.markdown("### Upload your PDF")
uploaded_pdf = st.file_uploader("Choose a PDF file to analyze", type=["pdf"], key="pdf_upload")

if uploaded_pdf:
    st.session_state.pdf_uploaded = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing your PDF..."):
        gpt.load_documents(tmp_path)
        st.success("PDF indexed successfully!")

    st.markdown("---")
    st.markdown("### Ask a Question")
    st.text_input("Type your question here:", key="user_question")

    if st.session_state.user_question.strip():
        with st.spinner("Thinking..."):
            answer = gpt.query(st.session_state.user_question)
        st.markdown("### Answer:")
        st.markdown(f"<div style='background-color: #3e4042; padding: 1rem; border-radius: 0.5rem;'>{answer}</div>", unsafe_allow_html=True)

        st.session_state.history.append(f"Q: {st.session_state.user_question}\nA: {answer}")
        st.session_state.user_question = ""  

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Q&A History")
        for qa in st.session_state.history:
            st.markdown(f"<div style='margin-bottom: 1rem; background-color: #313236; padding: 0.5rem; border-radius: 0.5rem;'><pre>{qa}</pre></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="📥 Download Q&A History",
                data="\n\n".join(st.session_state.history),
                file_name="qa_history.txt",
                mime="text/plain"
            )
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.history = []
                st.session_state.user_question = ""
                st.session_state.pdf_uploaded = False
                st.session_state.pdf_upload = None  
                st.experimental_rerun()
else:
    st.info("Please upload a PDF to begin.")
