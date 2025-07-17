import os
import streamlit as st
import pickle
import time
import re
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
file_path = "faiss_store_deepseek.pkl"
llm = ChatOllama(model="deepseek-r1:7b", base_url="http://127.0.0.1:11500")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

def extract_outside_think_tags(text):
    # Remove everything inside <think>...</think> including the tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

query = main_placeholder.text_input("Question: ")
if query:
    print(query,"recieved.")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            print("llm loaded data started")
            result = chain({"query": query}, return_only_outputs=True)
            result = result['result']
            print(result)
            outside_text = extract_outside_think_tags(result)
            st.header("Answer")
            st.write(outside_text)

