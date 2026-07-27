from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
import streamlit as st

@st.cache_resource
def cargar_manual_censo():
    print("Cargando y vectorizando el manual del Censo...")
    # 1. Cargar el PDF
    loader = PyPDFLoader("data/manual_uso_microdatos_censo2024.pdf")
    docs = loader.load()

    # 2. Cortar el texto en pedazos digeribles para el agente
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Vectorizar usando un modelo local super ligero y gratuito
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Crear la base de datos vectorial en memoria (FAISS)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 5. Configurar el recuperador (trae los 3 párrafos más relevantes)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Instanciamos el recuperador al iniciar la app
retriever_manual = cargar_manual_censo()

@tool
def consultar_manual_censo(query: str) -> str:
    """
    Usa esta herramienta EXCLUSIVAMENTE para buscar definiciones, fórmulas, 
    conceptos metodológicos o el significado de las variables del Censo 2024. 
    NO la uses para hacer cálculos ni para buscar datos numéricos.
    """
    resultados = retriever_manual.invoke(query)
    # Une los fragmentos encontrados en un solo texto para que el agente lo lea
    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    return f"Información del manual oficial:\n{contexto}"