import os
import sys
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
# 1. CONFIGURACIÓN CENTRALIZADA DE MODELOS
# ============================================================

# Modelos RECOMENDADOS para OpenRouter que SÍ soportan tool calling.
# Los modelos :free de OpenRouter TIENEN rate limits agresivos (1 req/seg)
# y a menudo NO soportan function calling correctamente.
# Si usas :free, considera agregar delays o usar modelos de pago.
MODELOS_NUBE_FALLBACK = [
    "anthropic/claude-3.5-haiku",           # Rápido, barato, buen tool calling
    "openai/gpt-4o-mini",                   # Excelente tool calling, económico
    "mistralai/mistral-small-3.1-24b-instruct",  # Buen balance
]

MODELO_LOCAL = "groq/llama-3.1-70b-versatile"


def _get_env(key: str, default=None):
    """Lee variable de entorno, con soporte para Streamlit secrets."""
    val = os.environ.get(key, default)
    # Intentar streamlit secrets solo si estamos en un contexto Streamlit
    if val is None:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            pass
    return val


def get_llms() -> List[ChatOpenAI]:
    """
    Devuelve una LISTA de instancias ChatOpenAI en orden de preferencia.
    Soporta: OpenRouter (nube) → OmniRoute (local).
    """
    openrouter_key = _get_env("OPENROUTER_API_KEY")
    openrouter_base = _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if openrouter_key:
        print("🌐 Modo activo: NUBE (OpenRouter) con fallback por reintento")
        # Headers requeridos por OpenRouter para evitar bloqueos
        extra_headers = {
            "HTTP-Referer": _get_env("APP_URL", "https://surdao.app"),
            "X-Title": "Sur DAO 2.0",
        }
        return [
            ChatOpenAI(
                model_name=m,                # ✅ model_name (no 'model')
                temperature=0,
                openai_api_key=openrouter_key,
                openai_api_base=openrouter_base,
                default_headers=extra_headers,  # ✅ Headers requeridos por OpenRouter
                max_retries=2,
            )
            for m in MODELOS_NUBE_FALLBACK
        ]
    else:
        # Modo local (OmniRoute u otro servidor OpenAI-compatible local)
        local_key = _get_env("LOCAL_API_KEY", "omniroute-local-key")
        local_base = _get_env("LOCAL_API_BASE", "http://localhost:20128/v1")

        # Verificación temprana: ¿responde el servidor local?
        try:
            import urllib.request
            req = urllib.request.Request(f"{local_base}/models", method="GET")
            req.add_header("Authorization", f"Bearer {local_key}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"💻 Modo activo: LOCAL ({local_base})")
                else:
                    print(f"⚠️ Servidor local respondió HTTP {resp.status}")
        except Exception as e:
            print(f"⚠️ Servidor local NO disponible en {local_base}: {e}")
            print("   → El agente fallará al invocarse. Revisa que OmniRoute esté corriendo.")

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
    """
    Determina si un error es FATAL (no tiene sentido reintentar con otro modelo)
    o TEMPORAL (vale la pena intentar con el siguiente modelo).
    """
    msg = str(error).lower()
    errores_fatales = [
        "authentication", "unauthorized", "invalid api key", "incorrect api key",
        "insufficient_quota", "billing", "payment", "rate limit exceeded",
        "invalid model", "model not found", "not a valid model",
    ]
    return any(e in msg for e in errores_fatales)


def construir_executors(tools, prompt):
    """Arma un AgentExecutor por cada modelo disponible."""
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
    """Prueba cada AgentExecutor en orden si ocurre un error temporal."""
    if not executors:
        return {"output": "⚠️ No hay modelos configurados disponibles. Revisa tu OPENROUTER_API_KEY o el servidor local."}

    ultimo_error = None
    for i, executor in enumerate(executors):
        try:
            return executor.invoke(input_dict)
        except Exception as e:
            ultimo_error = e
            if _es_error_fatal(e):
                print(f"🚫 Modelo #{i+1}: error FATAL ({type(e).__name__}). No se reintenta con él.")
                continue
            print(f"⚠️ Modelo #{i+1} falló ({type(e).__name__}: {str(e)[:150]}). Probando el siguiente...")
            continue

    return {
        "output": (
            f"⚠️ Todos los modelos disponibles fallaron.\n"
            f"Último error: {type(ultimo_error).__name__}: {str(ultimo_error)[:200]}\n\n"
            f"💡 Verifica:\n"
            f"  • Que OPENROUTER_API_KEY esté configurada (si usas nube)\n"
            f"  • Que OmniRoute esté corriendo en localhost:20128 (si usas local)\n"
            f"  • Que tu plan de OpenRouter tenga quota disponible"
        )
    }


# ============================================================
# 5. FÁBRICA DEL AGENTE PRINCIPAL
# ============================================================
def create_surdao_agent(dfs: dict):
    _python_tool = PythonAstREPLTool(locals={"pd": pd, "dfs": dfs})

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

    def generar_seccion_datamart(diccionario_tablas: dict) -> str:
        lineas = []
        for nombre, df in diccionario_tablas.items():
            columnas = ", ".join(df.columns[:8])
            lineas.append(f'- `dfs["{nombre}"]` → {df.shape[0]:,} filas. Columnas: {columnas}...')
        return "\n".join(lineas)

    seccion_datamart = generar_seccion_datamart(dfs)

    prompt_sistema = f"""Eres el **Agente Principal de Sur DAO**, un asistente experto en datos sociodemográficos, educativos y censales de Chile.

## 📁 DATAMART DISPONIBLE
Los nombres de tabla de abajo son EXACTOS — cópialos tal cual aparecen, entre comillas, sin traducirlos ni reformatearlos:
{seccion_datamart}

## 🔧 HERRAMIENTAS DISPONIBLES
1. **`ejecutar_pandas(codigo)`** → Para obtener, filtrar, cruzar y analizar datos. Siempre empieza con `df = dfs["Nombre EXACTO de la lista de arriba"]`.
2. **`buscar_tablas_en_datamart(palabra_clave)`** → SIEMPRE úsala primero si no estás 100% seguro del nombre exacto de una tabla, o si `ejecutar_pandas` te devuelve un KeyError. No adivines el nombre dos veces seguidas.
3. **`consultar_manual_censo(query)`** → Solo para definiciones metodológicas o fórmulas del Censo 2024.

## ⚠️ REGLAS ESTRICTAS
1. **NUNCA inventes el nombre de una tabla.** Si dudás, llama a `buscar_tablas_en_datamart` antes de `ejecutar_pandas`.
2. **Si `ejecutar_pandas` falla con KeyError**, tu próximo paso OBLIGATORIO es usar `buscar_tablas_en_datamart`.
3. **Filtrado de comunas:** usa `.str.lower()` en ambos lados. En `Histórico Educativo` la columna se llama `COMUNA` (mayúsculas).
4. **Sé eficiente:** Una vez que tengas los datos exactos, DETÉN el análisis y entrega la respuesta.
5. **Consultas complejas:** DIVIDE la respuesta en partes y guía al usuario paso a paso si el código es muy largo.
6. **SALUDOS Y BIENVENIDAS:** SOLO si el usuario te saluda de forma genérica ("hola", "buenos días") sin hacer ninguna pregunta de datos, responde con el mensaje de bienvenida. NUNCA repitas el saludo si el usuario ya está preguntando por una comuna o pidiendo información.
7. **PROHIBIDO DAR LISTAS GENÉRICAS O INVENTAR:** Si el usuario te pregunta "¿qué datos tienes de [Comuna]?", ESTÁ PROHIBIDO responder con una lista teórica de temas. Tienes que usar OBLIGATORIAMENTE `ejecutar_pandas` para extraer NÚMEROS REALES de esa comuna, y entregar la respuesta usando la estructura de formato. ¡Nunca inventes indicadores que no existan en el datamart!
8. MUCHAS tablas demográficas tienen una columna `sexo` con las categorías "Total", "Hombre" y "Mujer" en filas separadas.
   NUNCA sumes/agregues (`groupby().sum()`) sobre esta columna sin filtrar antes.
   Para el total de una comuna, filtrá explícitamente `df[df['sexo'] == 'Total']` (o el valor equivalente que tenga esa columna) ANTES de cualquier cálculo.
9. Una vez que tengas el DataFrame filtrado con los datos que necesitás, hacé TODOS los cálculos derivados (porcentajes, ratios) en el MISMO bloque de código, sobre esos datos reales. NUNCA vuelvas a escribir los números a mano en un DataFrame nuevo — eso es un paso extra innecesario y una fuente de errores de transcripción.

## 📋 REGLAS DE FORMATO PARA RESPUESTAS (OBLIGATORIO)
### 🔹 1. Resumen ejecutivo (máximo 3 líneas)
### 🔹 2. Tabla o lista de indicadores clave (máximo 5-6 filas)
### 🔹 3. Invitación a profundizar (opcional)

❌ LO QUE NO DEBES HACER:
- No incluyas datos crudos de todas las tablas consultadas.
- No uses formato científico.
- No mezcles markdown con texto sin formato.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    return construir_executors(herramientas, prompt)
