import os
import sys
import re
import unicodedata
import pandas as pd
import concurrent.futures
from typing import List

from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.documents import Document

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================
# 0. UTILIDAD DE NORMALIZACIÓN DE TEXTO (Tildes, Ñ y Caracteres Raros)
# ============================================================
def normalizar_texto_chile(texto: str) -> str:
    """
    Limpia tildes, mayúsculas y espacios extras, pero CONSERVA la letra 'ñ'.
    Ideal para emparejar nombres de comunas de forma robusta.
    """
    if not isinstance(texto, str):
        return str(texto)
    
    texto = texto.lower().strip()
    
    # Reemplazos seguros de vocales con tilde / diéresis
    reemplazos = [
        ('á','a'),('à','a'),('ä','a'),('â','a'),
        ('é','e'),('è','e'),('ë','e'),('ê','e'),
        ('í','i'),('ì','i'),('ï','i'),('î','i'),
        ('ó','o'),('ò','o'),('ö','o'),('ô','o'),
        ('ú','u'),('ù','u'),('ü','u'),('û','u')
    ]
    for orig, rem in reemplazos:
        texto = texto.replace(orig, rem)
        
    # Eliminar símbolos raros, puntuación o emojis (mantiene letras, números, espacios y la ñ)
    texto = re.sub(r'[^a-z0-9\sñ]', '', texto)
    # Compactar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto


# ============================================================
# 1. CONFIGURACIÓN CENTRALIZADA DE MODELOS
# ============================================================

MODELOS_NUBE_FALLBACK = [
    "anthropic/claude-3.5-haiku",           # Rápido, barato, buen tool calling
    "openai/gpt-4o-mini",                   # Excelente tool calling, económico
    "mistralai/mistral-small-3.1-24b-instruct",  # Buen balance
]

MODELO_LOCAL = "groq/openai/gpt-oss-120b"


def _get_env(key: str, default=None):
    """Lee variable de entorno, con soporte para Streamlit secrets."""
    val = os.environ.get(key, default)
    if val is None:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            pass
    return val


def get_llms() -> List[ChatOpenAI]:
    """Devuelve una LISTA de instancias ChatOpenAI en orden de preferencia."""
    openrouter_key = _get_env("OPENROUTER_API_KEY")
    openrouter_base = _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if openrouter_key:
        print("🌐 Modo activo: NUBE (OpenRouter) con fallback por reintento")
        extra_headers = {
            "HTTP-Referer": _get_env("APP_URL", "https://surdao.app"),
            "X-Title": "Sur DAO 2.0",
        }
        return [
            ChatOpenAI(
                model_name=m,
                temperature=0,
                openai_api_key=openrouter_key,
                openai_api_base=openrouter_base,
                default_headers=extra_headers,
                max_retries=2,
            )
            for m in MODELOS_NUBE_FALLBACK
        ]
    else:
        local_key = _get_env("LOCAL_API_KEY", "omniroute-local-key")
        local_base = _get_env("LOCAL_API_BASE", "http://localhost:20128/v1")

        try:
            import urllib.request
            req = urllib.request.Request(f"{local_base}/models", method="GET")
            req.add_header("Authorization", f"Bearer {local_key}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"💻 Modo activo: LOCAL ({local_base})")
        except Exception as e:
            print(f"⚠️ Servidor local NO disponible en {local_base}: {e}")

        return [
            ChatOpenAI(
                model_name=MODELO_LOCAL,
                temperature=0,
                openai_api_key=local_key,
                openai_api_base=local_base,
                max_retries=1,
            )
        ]


# ============================================================
# 2. RAG (MANUAL DEL CENSO)
# ============================================================
def get_rag_tool():
    docs = []
    pdf_path = "data/manual_uso_microdatos_censo2024.pdf"
    if os.path.exists(pdf_path):
        try:
            loader_pdf = PyPDFLoader(pdf_path)
            docs.extend(loader_pdf.load())
        except Exception as e:
            print(f"⚠️ Error cargando PDF: {e}")

    md_path = "data/columnas_totales.md"
    if os.path.exists(md_path):
        try:
            loader_md = TextLoader(md_path, encoding="utf-8")
            docs.extend(loader_md.load())
        except Exception as e:
            print(f"⚠️ Error cargando Markdown: {e}")

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @tool
    def consultar_manual_censo(query: str) -> str:
        """Úsala EXCLUSIVAMENTE para definiciones metodológicas, fórmulas o contexto del Censo 2024."""
        resultados = retriever.invoke(query)
        contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
        return f"📚 Información del Manual Censo 2024:\n{contexto}"

    return consultar_manual_censo


# ============================================================
# 3. ÍNDICE SEMÁNTICO DE TABLAS (BÚSQUEDA HÍBRIDA)
# ============================================================
def construir_indice_tablas(dfs: dict):
    documentos = []
    for nombre, df in dfs.items():
        columnas_texto = ", ".join(str(c) for c in df.columns)
        contenido = f"Tabla: {nombre}. Contiene las columnas: {columnas_texto}"
        documentos.append(
            Document(
                page_content=contenido,
                metadata={"nombre_tabla": nombre, "filas": df.shape[0], "columnas": list(df.columns)},
            )
        )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documentos, embeddings)


def crear_tool_busqueda_hibrida(dfs: dict, indice_semantico):
    @tool
    def buscar_tablas_en_datamart(consulta: str) -> str:
        """Encuentra qué tablas del datamart son relevantes para un tema."""
        consulta_lower = consulta.lower()
        exactas = []
        for nombre, df in dfs.items():
            if consulta_lower in nombre.lower() or any(consulta_lower in str(c).lower() for c in df.columns):
                exactas.append(f"📁 **{nombre}** ({df.shape[0]:,} filas) — coincidencia exacta")

        if exactas:
            return "🔎 Tablas encontradas (coincidencia exacta):\n" + "\n".join(exactas[:8])

        resultados = indice_semantico.similarity_search(consulta, k=5)
        if not resultados:
            return f"❌ No encontré tablas relacionadas con '{consulta}'."

        lineas = []
        for doc in resultados:
            nombre = doc.metadata["nombre_tabla"]
            filas = doc.metadata["filas"]
            columnas_muestra = ", ".join(doc.metadata["columnas"][:5])
            lineas.append(f"📁 **{nombre}** ({filas:,} filas) — columnas: {columnas_muestra}...")

        return "🔎 Tablas más relevantes (búsqueda semántica):\n" + "\n".join(lineas)

    return buscar_tablas_en_datamart


# ============================================================
# 4. FUNCIONES DE EJECUCIÓN CON FALLBACK
# ============================================================

def _es_error_fatal(error: Exception) -> bool:
    msg = str(error).lower()
    errores_fatales = [
        "authentication", "unauthorized", "invalid api key", "incorrect api key",
        "insufficient_quota", "billing", "payment", "rate limit exceeded",
        "invalid model", "model not found", "not a valid model",
    ]
    return any(e in msg for e in errores_fatales)


def construir_executors(tools, prompt):
    executors = []
    llms = get_llms()
    for i, llm in enumerate(llms):
        try:
            agente_base = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
            executor = AgentExecutor(
                agent=agente_base,
                tools=tools,
                verbose=True,
                max_iterations=25,
                handle_parsing_errors=True,
            )
            executors.append(executor)
        except Exception as e:
            print(f"⚠️ No se pudo construir executor con modelo #{i+1}: {e}")
    return executors


def invocar_con_fallback(executors: list, input_dict: dict) -> dict:
    if not executors:
        return {"output": "⚠️ No hay modelos configurados disponibles."}

    ultimo_error = None
    for i, executor in enumerate(executors):
        try:
            return executor.invoke(input_dict)
        except Exception as e:
            ultimo_error = e
            if _es_error_fatal(e):
                print(f"🚫 Modelo #{i+1}: error FATAL ({type(e).__name__}). No se reintenta.")
                continue
            print(f"⚠️ Modelo #{i+1} falló ({type(e).__name__}). Probando el siguiente...")
            continue

    return {
        "output": (
            f"⚠️ Todos los modelos disponibles fallaron.\n"
            f"Último error: {type(ultimo_error).__name__}: {str(ultimo_error)[:200]}"
        )
    }


# ============================================================
# 5. FÁBRICA DEL AGENTE PRINCIPAL
# ============================================================
def create_surdao_agent(dfs: dict):
    # Inyectamos la función de normalización en el entorno de ejecución de Pandas
    _python_tool = PythonAstREPLTool(locals={
        "pd": pd, 
        "dfs": dfs, 
        "normalizar": normalizar_texto_chile
    })

    @tool
    def ejecutar_pandas(codigo: str) -> str:
        """Ejecuta código Python/pandas contra el diccionario `dfs`."""
        def _ejecutar():
            return _python_tool.run(codigo)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futuro = executor.submit(_ejecutar)
            try:
                resultado = futuro.result(timeout=30)
                return str(resultado)[:4000]
            except concurrent.futures.TimeoutError:
                return "❌ ERROR: La consulta tomó más de 30 segundos. Divídela."
            except Exception as e:
                return f"❌ ERROR al ejecutar pandas: {type(e).__name__}: {str(e)[:500]}"

    indice_semantico_tablas = construir_indice_tablas(dfs)
    buscar_tablas_en_datamart = crear_tool_busqueda_hibrida(dfs, indice_semantico_tablas)

    herramientas = [ejecutar_pandas, buscar_tablas_en_datamart]
    rag_tool = get_rag_tool()
    if rag_tool:
        herramientas.append(rag_tool)

    prompt_sistema = """Eres el **Agente Principal de Sur DAO**, un asistente experto en datos sociodemográficos, educativos y censales de Chile, basado rigurosamente en el Manual del Censo 2024[cite: 5].

## 🔧 HERRAMIENTAS DISPONIBLES
1. **`buscar_tablas_en_datamart(palabra_clave)`** → Úsala primero para encontrar las tablas relevantes según el tema o comuna.
2. **`ejecutar_pandas(codigo)`** → Obligatoria para extraer las cifras reales. Empieza con `df = dfs["Nombre EXACTO"]`.
3. **`consultar_manual_censo(query)`** → Solo para definiciones metodológicas o fórmulas[cite: 5].

## ⚠️ REGLAS ESTRICTAS Y METODOLÓGICAS (MANUAL CENSO 2024)[cite: 5]
1. **PROHIBIDO SER UN AGENTE VAGO:** Si usas `buscar_tablas_en_datamart`, TIENES PROHIBIDO limitarte a mostrar nombres de tablas al usuario. Debes invocar inmediatamente `ejecutar_pandas` para extraer los números y presentar las cifras reales.
2. **CERO INVENTOS:** Si un dato no está en el datamart, di "No disponible"[cite: 4, 6]. Está prohibido usar datos de ejemplo.
3. **Manejo de Valores Especiales:** Antes de promediar o sumar, DEBES excluir los valores especiales: `-99` (No responde), `-66` (Suprimido por anonimización) y `NA` (No aplica)[cite: 5, 6].
4. **Cálculo de Proporciones:** Excluye siempre los casos de "No respuesta" (`-99`) del denominador[cite: 5, 6].
5. **Filtrado Robusto de Comunas y Textos:** Como pueden haber variaciones por tildes o mayúsculas, utiliza la función auxiliar disponible `normalizar()` sobre las columnas de texto antes de filtrar (ej: `df[df['comuna'].apply(normalizar) == normalizar('Valparaíso')]`). Respeta y conserva siempre la letra `ñ` (ej: `ñuñoa`).
6. **Filtro de Sexo y Totales:** Las tablas demográficas separan las filas por `sexo` ("Total", "Hombre", "Mujer"). NUNCA sumes sin filtrar antes explícitamente `df[df['sexo'] == 'Total']` (o equivalente) para evitar duplicar población.
7. **Redondeo:** Todos los indicadores y promedios finales deben presentarse redondeados a un (1) decimal[cite: 6].

## 📋 REGLAS DE FORMATO PARA RESPUESTAS (OBLIGATORIO)
### 🔹 1. Resumen ejecutivo (máximo 3 líneas)
### 🔹 2. Tabla o lista de indicadores clave (máximo 5-6 filas)
### 🔹 3. Invitación a profundizar (opcional)
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    return construir_executors(herramientas, prompt)