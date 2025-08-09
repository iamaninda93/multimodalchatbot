import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PIL import Image
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os

# 🔐 Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# ⚡ Gemini multimodal model
model = genai.GenerativeModel("gemini-1.5-flash")

# 📄 Sidebar for file upload
st.sidebar.title("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload text, PDF, or image files", type=["txt", "pdf", "jpg", "png"], accept_multiple_files=True
)

# 📚 Document store setup
documents = []
images = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type

        # 📝 Handle text file
        if file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")

        # 📕 Handle PDF file
        elif file_type == "application/pdf":
            pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in pdf_reader:
                text += page.get_text()

        # 🖼️ Handle image file
        elif file_type.startswith("image"):
            image = Image.open(uploaded_file)
            images.append(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            continue  # Skip text processing for images

        # 🔍 Embed and store text chunks
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        documents.extend([Document(page_content=chunk) for chunk in chunks])

    # 🧠 Create vector store
    if documents:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(documents, embeddings)

# 💬 User query
query = st.text_input("Ask a question about your uploaded content")

# 🤖 RAG + Gemini response
if query:
    context = ""
    if documents:
        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

    # 🧠 Multimodal input: text + images
    inputs = [query]
    if context:
        inputs.append(context)
    inputs.extend(images)

    response = model.generate_content(inputs)

    st.markdown("### ✨ Gemini's Response")
    st.write(response.text)
