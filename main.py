'''
tut: https://medium.com/@diptimanrc/rapid-q-a-on-multiple-pdfs-using-langchain-and-chromadb-as-local-disk-vector-store-60678328c0df
'''

# import(‘pysqlite3’) import sys sys.modules[‘sqlite3’] = sys.modules.pop(‘pysqlite3’)
import sys
sys.modules['sqlite3'] = __import__('pysqlite3')

import streamlit as st

# from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
# from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader

from langchain.document_loaders import PyPDFLoader, PythonLoader, UnstructuredWordDocumentLoader#, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
# # # from langchain.vectorstores import Chroma # from tut
from langchain.vectorstores.chroma import Chroma #https://github.com/langchain-ai/langchain/issues/7119
import chromadb

import os
# import toml
# os.environ['OPENAI_API_KEY'] = toml.get(OPENAI_API_KEY)

# from credentials.api.openai import OPENAI_API_KEY
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from dotenv import load_dotenv
def configure():
    load_dotenv()

# # # from os import listdir
# # # from os.path import isfile, join

def load_chunk_persist_pdf() -> Chroma:
    folder_path = "C:\\Users\\me\\Projects\\llm\\langchain-chromadb-pdf-2\\data"
    documents = []
    # for file in os.listdir(folder_path):
    # data_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    # for file in data_files:
    for root, dirs, files in os.walk(folder_path):
        for file in files:    
            if file.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            if file.endswith('.py'):
                pdf_path = os.path.join(root, file)
                loader = PythonLoader(pdf_path)
                documents.extend(loader.load())
            if file.endswith('.doc'):
                pdf_path = os.path.join(folder_path, file)
                loader = UnstructuredWordDocumentLoader(pdf_path)
                documents.extend(loader.load())
            if file.endswith('.docx'):
                pdf_path = os.path.join(folder_path, file)
                loader = UnstructuredWordDocumentLoader(pdf_path) #Docx2txtLoader
                documents.extend(loader.load())
            # if file.endswith('.txt'):
            #     pdf_path = os.path.join(folder_path, file)
            #     loader = TextLoader(pdf_path)
            #     documents.extend(loader.load())
            # if file.endswith('.md'):
            #     pdf_path = os.path.join(folder_path, file)
            #     loader = UnstructuredMarkdownLoader(pdf_path)
            #     documents.extend(loader.load())

    # ".txt": TextLoader,
    # ".md": UnstructuredMarkdownLoader,
    # ".py": TextLoader,
    # # ".pdf": PDFMinerLoader,
    # ".pdf": UnstructuredFileLoader,
    # ".csv": CSVLoader,
    # ".xls": UnstructuredExcelLoader,
    # ".xlsx": UnstructuredExcelLoader,
    # ".docx": Docx2txtLoader,
    # ".doc": Docx2txtLoader,
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("dsgpt_collection_a3")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=r"./db"
    )
    vectordb.persist()
    return vectordb 

def create_agent_chain():
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    configure()
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="DSGPT", page_icon=":robot:")
st.header("DSGPT: Ask python libraries how they work")

form_input = st.text_input('Enter Question and Click Enter with your Mouse')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))
