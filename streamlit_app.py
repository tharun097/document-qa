import streamlit as st
from groq import Groq
import tempfile
from PyPDF2 import PdfReader

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
st.set_page_config(page_title="Document Q&A", page_icon="📄")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------------------------
# UI
# -----------------------------------------------
st.title("📄 Document Q&A — powered by Groq")
st.write("Upload a document and ask any question about it.")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "md"])
document_text = ""


# -----------------------------------------------
# LOAD DOCUMENT
# -----------------------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    if uploaded_file.name.lower().endswith(".pdf"):
        pdf = PdfReader(temp_path)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            text += extracted
        document_text = text

    else:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            document_text = f.read()

    st.success(f"📄 {uploaded_file.name} uploaded successfully!")

    with st.expander("📄 Show Document Preview"):
        st.text_area(
            "Document Preview",
            document_text[:4000],
            height=250
        )


# -----------------------------------------------
# ASK QUESTION
# -----------------------------------------------
question = st.text_input(
    "💬 Ask your question:",
    placeholder="Example: Summarize the document...",
    disabled=not uploaded_file
)

# -----------------------------------------------
# STREAMING ANSWER (CLEAN TEXT ONLY)
# -----------------------------------------------
if uploaded_file and question:
    st.subheader("🤖 Answer")

    prompt = f"""
You are an AI assistant. Use the document below to answer the question.

DOCUMENT:
-------------------
{document_text}
-------------------

QUESTION:
{question}

Give a clear and concise answer.
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        stream=True
    )

    # Stream clean text only
    def stream_text(chunks):
        for chunk in chunks:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content

    st.write_stream(stream_text(stream))
