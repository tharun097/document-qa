import streamlit as st
from groq import Groq
import tempfile

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
st.set_page_config(page_title="Document Q&A (Groq)", page_icon="📄")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------------------------
# UI
# -----------------------------------------------
st.title("📄 Chat with Your Documents — Powered by Groq 🚀")
st.write("Upload a document and ask questions about its content.")

# -----------------------------------------------
# FILE UPLOAD
# -----------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md"],
)

# -----------------------------------------------
# READ DOCUMENT CONTENT
# -----------------------------------------------
document_text = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # PDF Handling
    if uploaded_file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        pdf = PdfReader(temp_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        document_text = text

    # TXT or MD
    else:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            document_text = f.read()

    st.success(f"📄 Document loaded successfully: {uploaded_file.name}")
    st.subheader("📘 Extracted Document Preview")
    st.text_area("Document Content", document_text[:3000], height=200)

# -----------------------------------------------
# QUESTION INPUT
# -----------------------------------------------
question = st.text_area(
    "💬 Ask a question about the document:",
    placeholder="E.g., Summarize the key points...",
    disabled=not uploaded_file
)

# -----------------------------------------------
# HANDLE QUESTION + STREAMING ANSWER
# -----------------------------------------------
if uploaded_file and question:

    st.subheader("🤖 Answer")
    full_prompt = f"""
You are a helpful assistant. Use the document below to answer the question.

DOCUMENT:
---------------------
{document_text}
---------------------

QUESTION:
{question}

Answer in a clear and concise way.
"""

    # Prepare messages
    messages = [
        {"role": "user", "content": full_prompt}
    ]

    # Stream response
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True
    )

    response_container = st.empty()
    st.write_stream(stream)
