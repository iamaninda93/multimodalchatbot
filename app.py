import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Use it with Google Generative AI
genai.configure(api_key=api_key)

# 🧠 Load Gemini model

#model = genai.GenerativeModel("gemini-1.5-pro")
model = genai.GenerativeModel("gemini-1.5-flash")



# 📄 Sidebar for file upload
st.sidebar.title("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload a text or image file", type=["txt", "jpg", "png"])

# 📚 Document store setup
documents = []
if uploaded_file and uploaded_file.type == "text/plain":
    text = uploaded_file.read().decode("utf-8")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embeddings)

# 🖼️ Image handling
if uploaded_file and uploaded_file.type.startswith("image"):
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# 💬 User query
query = st.text_input("Ask a question about your uploaded content")

# 🔍 RAG + Gemini response
if query:
    context = ""
    if documents:
        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

    if uploaded_file and uploaded_file.type.startswith("image"):
        response = model.generate_content([query, image])
    else:
        response = model.generate_content(f"{context}\n\nQuestion: {query}")

    st.markdown("### ✨ Gemini's Response")
    st.write(response.text)
