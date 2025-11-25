import streamlit as st
import tempfile
import docx
import PyPDF2
from langchain_groq import ChatGroq

# ---------------------------------------------------------
# INTERNAL GROQ API KEY
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("📄 Document Question Answering (Groq LLM)")
st.write("Upload a document and ask questions about it using Groq Llama 3.")

uploaded_file = st.file_uploader(
    "Upload a document (.pdf, .txt, .md, .docx)",
    type=["pdf", "txt", "md", "docx"]
)

question = st.text_area(
    "Ask a question about the document",
    placeholder="Example: Give me a short summary..."
)

# ---------------------------------------------------------
# READ DOCUMENT (NO PDFPLUMBER)
# ---------------------------------------------------------
def read_document(file):
    filename = file.name.lower()

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        temp_path = tmp.name

    # PDF → text using PyPDF2
    if filename.endswith(".pdf"):
        text = ""
        with open(temp_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    # DOCX → text
    if filename.endswith(".docx"):
        doc = docx.Document(temp_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # TXT / MD
    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------
# PROCESS INPUT & SEND TO GROQ
# ---------------------------------------------------------
if uploaded_file and question:

    document_text = read_document(uploaded_file)

    st.subheader("📤 Your Question")
    st.write(question)

    client = ChatGroq(api_key=GROQ_API_KEY)

    # Prepare Groq messages
    messages = [
        {
            "role": "user",
            "content": (
                f"Here is the document:\n\n{document_text}\n\n---\n\n"
                f"Question: {question}"
            )
        }
    ]

    st.subheader("🤖 Response")

    # Stream Groq LLM output
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)
