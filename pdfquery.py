# cd D:\ai-project
# streamlit run visual_path\project\pdfquery.py

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---------------------------------------------------
# Load Environment
# ---------------------------------------------------

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get(
    "OPENAI_API_KEY", None
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

os.makedirs(PDF_DIR, exist_ok=True)

# ---------------------------------------------------
# Create ONE embedding object only
# ---------------------------------------------------

EMBEDDINGS = OpenAIEmbeddings(
    api_key=OPENAI_KEY,
    model="text-embedding-3-large"
)

# ---------------------------------------------------
# Save Uploaded PDFs
# ---------------------------------------------------

def save_uploaded_files(uploaded_files):
    paths = []

    for uploaded_file in uploaded_files:
        path = os.path.join(PDF_DIR, uploaded_file.name)

        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        paths.append(path)

    return paths

# ---------------------------------------------------
# Extract PDF Text
# ---------------------------------------------------

def get_pdf_text(pdf_paths):

    text = ""

    for pdf_path in pdf_paths:

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            try:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            except Exception as e:
                print(e)

    return text

# ---------------------------------------------------
# List PDFs
# ---------------------------------------------------

def list_stored_pdfs():
    return [
        f for f in os.listdir(PDF_DIR)
        if os.path.isfile(os.path.join(PDF_DIR, f))
    ]

# ---------------------------------------------------
# Split Text
# ---------------------------------------------------

def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)

# ---------------------------------------------------
# Build Vector Store
# ---------------------------------------------------

def get_vector_store(text_chunks):

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=EMBEDDINGS
    )

    vector_store.save_local(FAISS_DIR)

# ---------------------------------------------------
# QA Chain
# ---------------------------------------------------

def get_conversational_chain():

    prompt_template = """
You are an AI PDF expert.

Answer only from the provided context.

If the answer is unavailable, reply:

"Answer is not available in the provided PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatOpenAI(
        api_key=OPENAI_KEY,
        model="gpt-4.1-mini",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

# ---------------------------------------------------
# User Question
# ---------------------------------------------------

def user_input(user_question):

    if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):

        st.warning("Please upload and process PDFs first.")

        return

    db = FAISS.load_local(
        FAISS_DIR,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(
        user_question,
        k=4
    )

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    st.write("### Reply")

    st.write(response["output_text"])

# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    st.set_page_config(
        page_title="Chat with PDF",
        layout="wide"
    )

    st.title("📄 Chat with Your PDFs")

    with st.sidebar:

        st.header("Upload PDFs")

        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process Uploaded PDFs"):

            if not uploaded_files:

                st.warning("Please upload at least one PDF.")

            else:

                with st.spinner("Processing PDFs..."):

                    pdf_paths = save_uploaded_files(uploaded_files)

                    raw_text = get_pdf_text(pdf_paths)

                    if raw_text.strip() == "":

                        st.error("No text found in uploaded PDFs.")

                    else:

                        text_chunks = get_text_chunks(raw_text)

                        get_vector_store(text_chunks)

                        st.success("PDFs processed successfully!")

        st.divider()

        st.subheader("Stored PDFs")

        pdfs = list_stored_pdfs()

        if pdfs:

            for pdf in pdfs:
                st.write(pdf)

        else:

            st.write("No PDFs uploaded.")

    st.header("Ask a Question")

    question = st.text_input(
        "Type your question"
    )

    if question:

        user_input(question)

# ---------------------------------------------------

if __name__ == "__main__":
    main()