import streamlit as st
import pdfplumber
import google.generativeai as genai
import numpy as np
import faiss
import cv2

from PIL import Image

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# ===================================
# GEMINI API
# ===================================

import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)
# ===================================
# CUSTOM CSS
# ===================================

st.markdown("""
<style>

.stApp{
    background:#f8fafc;
}

.main-title{
    text-align:center;
    color:#2563eb;
    font-size:50px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#475569;
    font-size:18px;
}

.answer-box{
    background:white;
    padding:20px;
    border-radius:10px;
    border:1px solid #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ===================================
# HEADER
# ===================================

st.markdown(
"""
<h1 class='main-title'>
🤖 AI Document Assistant
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<p class='subtitle'>
Upload a PDF and ask questions about it
</p>
""",
unsafe_allow_html=True
)

st.divider()


# ===================================
# PDF TEXT EXTRACTION
# ===================================

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# ===================================
# TEXT CLEANING
# ===================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


# ===================================
# TEXT CHUNKING
# ===================================

def create_chunks(
    text,
    chunk_size=500
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks


# ===================================
# DOCUMENT STATS
# ===================================

def get_document_stats(text):

    words = len(
        text.split()
    )

    chars = len(
        text
    )

    return words, chars


# ===================================
# OPENCV IMAGE PROCESSING
# ===================================

def preprocess_image(image):

    image = np.array(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


# ===================================
# DISPLAY DOCUMENT INFO
# ===================================

def show_document_info(
    words,
    chars
):

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "📝 Words",
            f"{words:,}"
        )

    with col2:

        st.metric(
            "📄 Characters",
            f"{chars:,}"
        )

# ===================================
# PDF TEXT EXTRACTION
# ===================================

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# ===================================
# TEXT CLEANING
# ===================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


# ===================================
# TEXT CHUNKING
# ===================================

def create_chunks(
    text,
    chunk_size=500
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks


# ===================================
# DOCUMENT STATS
# ===================================

def get_document_stats(text):

    words = len(
        text.split()
    )

    chars = len(
        text
    )

    return words, chars


# ===================================
# OPENCV IMAGE PROCESSING
# ===================================

def preprocess_image(image):

    image = np.array(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


# ===================================
# DISPLAY DOCUMENT INFO
# ===================================

def show_document_info(
    words,
    chars
):

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "📝 Words",
            f"{words:,}"
        )

    with col2:

        st.metric(
            "📄 Characters",
            f"{chars:,}"
        )

# ===================================
# SIMPLE EMBEDDINGS
# ===================================

def text_to_vector(text):

    vector = np.zeros(128)

    for i, char in enumerate(text[:128]):

        vector[i] = ord(char)

    return vector.astype("float32")


# ===================================
# BUILD FAISS INDEX
# ===================================

def build_faiss(chunks):

    vectors = np.array(

        [
            text_to_vector(chunk)

            for chunk in chunks
        ]

    )

    index = faiss.IndexFlatL2(128)

    index.add(vectors)

    return index


# ===================================
# RETRIEVE RELEVANT TEXT
# ===================================

def retrieve_relevant_chunks(

    question,
    chunks,
    index

):

    query_vector = text_to_vector(
        question
    )

    distances, indices = index.search(

        np.array(
            [query_vector]
        ),

        k=3

    )

    context = ""

    for idx in indices[0]:

        if idx < len(chunks):

            context += chunks[idx]

            context += "\n\n"

    return context


# ===================================
# PROMPT ENGINEERING
# ===================================

def create_prompt(

    context,
    question

):

    prompt = f"""

You are an AI Document Assistant.

Answer ONLY using the document content.

Rules:

1. Give concise answers.

2. Use bullet points when possible.

3. If the answer is unavailable,
   say:

   Information not found in document.

Document Context:

{context}

Question:

{question}

Answer:

"""

    return prompt


# ===================================
# GEMINI RESPONSE
# ===================================

def ask_gemini(

    context,
    question

):

    prompt = create_prompt(

        context,
        question

    )

    response = model.generate_content(
        prompt
    )

    return response.text


# ===================================
# DOCUMENT PROCESSING PIPELINE
# ===================================

def process_document(pdf_file):

    text = extract_text_from_pdf(
        pdf_file
    )

    text = clean_text(
        text
    )

    chunks = create_chunks(
        text
    )

    index = build_faiss(
        chunks
    )

    words, chars = get_document_stats(
        text
    )

    return (

        text,

        chunks,

        index,

        words,

        chars

    )


# ===================================
# ANSWER DISPLAY
# ===================================

def show_answer(answer):

    st.info("🤖 AI Answer")

    st.markdown(answer)

# ===================================
# FILE UPLOAD SECTION
# ===================================

st.subheader("📄 Upload PDF")

pdf_file = st.file_uploader(
    "Choose a PDF File",
    type=["pdf"]
)

st.subheader("📷 Upload Image (Optional)")

image_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ===================================
# IMAGE PROCESSING
# ===================================

if image_file:

    image = Image.open(image_file)

    processed_image = preprocess_image(
        image
    )

    st.image(
        processed_image,
        caption="Processed Image",
        use_container_width=True
    )

# ===================================
# PDF PROCESSING
# ===================================

if pdf_file:

    with st.spinner(
        "Processing PDF..."
    ):

        text, chunks, index, words, chars = process_document(
            pdf_file
        )

    st.success(
        "PDF Processed Successfully"
    )

    show_document_info(
        words,
        chars
    )

    st.subheader(
        "💬 Ask Question"
    )

    question = st.text_input(
        "Ask anything about your document"
    )

    if st.button(
        "Get Answer"
    ):

        context = retrieve_relevant_chunks(
            question,
            chunks,
            index
        )

        answer = ask_gemini(
            context,
            question
        )

        show_answer(
            answer
        )