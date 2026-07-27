import os
import signal
import pandas as pd
import concurrent.futures
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================
# 1. CONFIGURACIÓN DEL LLM (LOCAL / NUBE)
# ============================================================
def get_llm():
    """Detecta si estamos en local (Omniroute) o en la nube (OpenRouter)"""
    from dotenv import load_dotenv
    load_dotenv()

    usar_nube = False  # 🔥 Cambia a True si quieres usar OpenRouter

    if usar_nube and os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        modelo_activo = "google/gemma-2-9b-it:free" 
    else:
        # 🔥 OmniRoute local
        os.environ["OPENAI_API_KEY"] = "omniroute-local-key"
        os.environ["OPENAI_API_BASE"] = "http://localhost:20128/v1"
        modelo_activo = "oc/deepseek-v4-flash-free" 

    return ChatOpenAI(model=modelo_activo, temperature=0)

# ============================================================
# 2. RAG (MANUAL DEL CENSO)
# ============================================================
def get_rag_tool():
    """Construye la herramienta de búsqueda en el manual del Censo 2024"""
    docs = []

    # 1. Manual en PDF
    pdf_path = "data/manual_uso_microdatos_censo2024.pdf"
    if os.path.exists(pdf_path):
        try:
            loader_pdf = PyPDFLoader(pdf_path)
            docs.extend(loader_pdf.load())
            print(f"✅ PDF cargado: {pdf_path}")
        except Exception as e:
            print(f"⚠️ Error cargando PDF: {e}")

    # 2. Diccionario de columnas en Markdown
    md_path = "data/columnas_totales.md"
    if os.path.exists(md_path):
        try:
            loader_md = TextLoader(md_path, encoding="utf-8")
            docs.extend(loader_md.load())
            print(f"✅ Markdown cargado: {md_path}")
        except Exception as e:
            print(f"⚠️ Error cargando Markdown: {e}")

    if not docs:
        print("⚠️ No se encontraron documentos para RAG.")
        return None

    # Chunking y vectorización
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @tool
    def consultar_manual_censo(query: str) -> str:
        """Úsala EXCLUSIVAMENTE para definiciones metodológicas, fórmulas o contexto del Censo 2024. NO la uses para obtener datos, para eso usa 'ejecutar_pandas'."""
        resultados = retriever.invoke(query)
        contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
        return f"📚 Información del Manual Censo 2024:\n{contexto}"

    return consultar_manual_censo

# ============================================================
# 3. FÁBRICA DEL AGENTE PRINCIPAL
# ============================================================
def create_surdao_agent(dfs: dict):
    """Construye el agente con todas las herramientas y el prompt optimizado."""
    llm = get_llm()

    # --- Tool 1: Ejecutor de pandas (con timeout robusto para Windows/Linux) ---
    _python_tool = PythonAstREPLTool(locals={"pd": pd, "dfs": dfs})

    @tool
    def ejecutar_pandas(codigo: str) -> str:
        """
        Ejecuta código Python/pandas contra el diccionario `dfs`.
        Úsala para obtener, filtrar, cruzar y analizar datos del Censo.
        El código debe ser autónomo y usar `df = dfs["Nombre exacto de la tabla"]`.
        """
        def _ejecutar():
            return _python_tool.run(codigo)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futuro = executor.submit(_ejecutar)
            try:
                resultado = futuro.result(timeout=30)
                return str(resultado)[:4000]
            except concurrent.futures.TimeoutError:
                return "❌ ERROR: La consulta tomó más de 30 segundos. Divídela en partes más pequeñas."
            except Exception as e:
                return f"❌ ERROR al ejecutar pandas: {type(e).__name__}: {str(e)[:500]}"

    # --- Tool 2: Buscador de tablas en el datamart ---
    @tool
    def buscar_tablas_en_datamart(palabra_clave: str) -> str:
        """
        BÚSQUEDA INTELIGENTE: encuentra qué tablas del datamart contienen información sobre un tema específico.
        Ejemplos: 'fecundidad', 'religión', 'migración', 'discapacidad', 'envejecimiento', 'educación'.
        Úsala cuando no sepas exactamente qué tabla contiene los datos que necesitas.
        """
        encontradas = []
        for nombre, df in dfs.items():
            coincide_nombre = palabra_clave.lower() in nombre.lower()
            coincide_columna = any(palabra_clave.lower() in col.lower() for col in df.columns)

            if coincide_nombre or coincide_columna:
                prioridad = 0 if coincide_nombre else 1
                columnas = list(df.columns)[:5]
                filas = df.shape[0]
                encontradas.append((prioridad, 
                    f"📁 **{nombre}** ({filas:,} filas)\n"
                    f"   - Columnas: {columnas}...\n"))

        if encontradas:
            encontradas.sort(key=lambda x: x[0])  # Prioridad: nombre primero
            return "🔎 Tablas encontradas en el datamart:\n" + "\n".join(e[1] for e in encontradas[:8])
        return f"❌ No encontré tablas relacionadas con '{palabra_clave}'."

    # --- Ensamblaje de Herramientas ---
    herramientas = [ejecutar_pandas, buscar_tablas_en_datamart]
    rag_tool = get_rag_tool()
    if rag_tool:
        herramientas.append(rag_tool)

    # --- PROMPT DEL SISTEMA ---
    prompt_sistema = """Eres el **Agente Principal de Sur DAO**, un asistente experto en datos sociodemográficos, educativos y censales de Chile.

## 📁 DATAMART DISPONIBLE
Tienes acceso a un diccionario llamado `dfs` con las siguientes tablas (usa `buscar_tablas_en_datamart` para explorar temas no listados):

### Educacionales e Históricos
- `dfs["Histórico Educativo (2012-2023)"]` → Columnas: RBD, Nombre_Colegio, Total_Alumnos, Total_Docentes, Ratio_Alumnos_Docente, Promedio_Notas, Anio, COMUNA, REGION, Volatilidad_Rendimiento
- `dfs["Matriz 2024 + Censo"]` → Columnas: RBD, Nombre_Colegio, Total_Alumnos, Ratio_Alumnos_Docente, Promedio_Notas, comuna, sabe_leer_y_escribir, poblacion_de_5_años_o_más, parvularia, básica, media, superior, tasa_de_asistencia_neta_educacion_básica_/r
- `dfs["Años Escolaridad (P7_4)"]` → comuna, sexo, años_de_escolaridad_promedio, años_de_escolaridad_promedio_para_la_poblacion_de_18_años_o_más
- `dfs["Asistencia Neta Comunal (P7_8)"]` → comuna, tasa_de_asistencia_neta_educacion_parvularia_/r, ...básica_/r, ...media_/r, ...superior_/r

### Demografía y Diversidad
- `dfs["Envejecimiento (D2_2)"]` → comuna, sexo, poblacion_censada, 0_14, 15_64, 65_años_o_más, indice_de_envejecimiento
- `dfs["Discapacidad (P1_2)"]` → comuna, grupos_de_edad, poblacion_de_5_años_o_más_con_discapacidad, hombre, mujer
- `dfs["Pueblos Originarios (P2_2)"]` → comuna, poblacion_que_es_o_se_considera_perteneciente_a_un_pueblo_indigena_u_originario, mapuche, aymara, rapa_nui, atacameño_o_lickanantay, quechua, colla, diaguita, kawésqar, yagán, chango, selk'nam, otro, pueblo_no_declarado
- `dfs["Alfabetización Comunal (P7_10)"]` → comuna, sabe_leer_y_escribir, poblacion_de_5_años_o_más, 5_14_años, 15_64_años, 65_años_o_más
- `dfs["Nivel Educativo Comunal (P7_2)"]` → comuna, poblacion_censada, nunca_asistio, diferencial, parvularia, básica, media, superior, nivel_educativo_no_declarado

### Migración y Movilidad
- `dfs["Escolaridad Inmigrantes (P8_2)"]` → comuna, años_de_escolaridad_promedio, años_de_escolaridad_promedio_para_la_poblacion_de_18_años_o_más
- `dfs["Inmigrantes por País (D4_4)"]` → comuna, pais_o_continente_de_nacimiento, inmigrantes_internacionales
- `dfs["Migración Interna (D5_2)"]` → ESTRUCTURA ANCHA: 'comuna_de_residencia_habitual_actual', 'poblacion_censada', 'no_migrante_interno_comunal', 'aún_no_nacian_(menores_de_5_años)' + una columna por cada comuna de Chile con el número de personas llegadas desde allí.

## 🔧 HERRAMIENTAS DISPONIBLES
1. **`ejecutar_pandas(codigo)`** → Para obtener, filtrar, cruzar y analizar datos. Siempre empieza con `df = dfs["Nombre exacto de la tabla"]`.
2. **`buscar_tablas_en_datamart(palabra_clave)`** → Para descubrir qué tablas contienen un tema específico.
3. **`consultar_manual_censo(query)`** → Solo para definiciones metodológicas o fórmulas del Censo 2024.

## ⚠️ REGLAS ESTRICTAS
1. **NUNCA inventes datos.** Siempre usa `ejecutar_pandas` para obtenerlos.
2. **Filtrado de comunas:** usa `.str.lower()` en ambos lados:
   - `df[df['comuna'].str.lower() == 'sierra gorda']`
   - En `Histórico Educativo` la columna se llama `COMUNA` (mayúsculas).
3. **Sé eficiente:** Una vez que tengas los datos exactos que responden la pregunta, DETÉN el análisis y entrega la respuesta. No hagas comprobaciones adicionales.
4. **Consultas complejas:** Si necesitas cruzar más de 2 tablas o el código es muy largo, DIVIDE la respuesta en partes y guía al usuario paso a paso.
5. **Errores:** Si encuentras un error de timeout o límite de iteraciones, responde con un mensaje amigable pidiendo al usuario que simplifique o divida la pregunta.
6. **SALUDOS Y BIENVENIDAS:** Si el usuario te saluda ("hola", "buenos días") o te pregunta qué puedes hacer, responde con un mensaje de bienvenida y una breve descripción de tus capacidades. Responde SIEMPRE invitándolo a explorar de lo general a lo particular con este ejemplo:
   🤖 ¡Hola! Soy **Sur DAO 2.0**, tu Agente Analítico Especializado en Educación y Sociodemografía de Chile. 🇨🇱📊\n\n"
        "Tengo en mi memoria un datamart avanzado con más de **60 bases de datos cruzadas** (Censo 2024 y MINEDUC).\n\n"
        "💡 **¿Cómo sacarme el mayor provecho?**\n"
        "Te recomiendo ir de lo general a lo particular. Aquí tienes un flujo que funciona perfecto:\n\n"
        "1️⃣ **Parte por tu comuna:**\n"
        "👉 *'¿Qué datos tienes de Talagante?'*\n\n"
        "2️⃣ **Haz zoom en un tema:**\n"
        "👉 *'¿Cómo es la educación de los inmigrantes comparada con los nacionales ahí?'*\n\n"
        "3️⃣ **Cruza con evolución histórica:**\n"
        "👉 *'Respecto a la evolución del rendimiento y el ratio estudiantes/profesor, ¿qué datos históricos tienes?'*\n\n"
        "También puedes usar `/buscar [tema]` para encontrar tablas en mi base de datos.\n\n"
        "¿Por qué comuna empezamos a analizar hoy? 🚀

## 📋 REGLAS DE FORMATO PARA RESPUESTAS (OBLIGATORIO)

Sigue esta estructura **siempre** que respondas con datos:

### 🔹 1. Resumen ejecutivo (máximo 3 líneas)
Entrega los datos más impactantes de forma concisa. Ejemplo:
- La comuna de Sierra Gorda tiene una población censada de 1.472 personas.
- El ratio de alumnos por docente en la comuna es de 15:1.
- El promedio de notas en la comuna es de 6.5.

### 🔹 2. Tabla o lista de indicadores clave (máximo 5-6 filas)
Selecciona solo los indicadores más relevantes para la pregunta. Usa formato tabla o viñetas simples.

### 🔹 3. Invitación a profundizar (opcional)
Termina con una pregunta abierta para que el usuario pueda elegir el siguiente paso.

### ❌ LO QUE NO DEBES HACER:
- No incluyas datos crudos de todas las tablas consultadas.
- No incluyas listas largas de países, colegios o grupos de edad completos.
- No uses formato científico (ej: `1.0e3`). Usa números enteros o con 1 decimal.
- No mezcles markdown con texto sin formato (evita `_`, `*`, `[` sueltos).
- Si el usuario pregunta por una comuna, no le entregues las 12 tablas. Elige los 5-6 datos más relevantes.

✅ **Ejemplo de respuesta ideal:**

🏔️ Sierra Gorda — 1.472 hab.

- Índice de envejecimiento: 27.9 (población muy joven)
- 31.8% inmigrantes | 14.7% pueblos originarios
- Escolaridad 18+: 11.2 años | Asistencia básica: 95.4%

¿Quieres que profundice en educación, migración o demografía?

"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sistema),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- CONSTRUCCIÓN DEL AGENTE ---
    agente_base = create_tool_calling_agent(llm=llm, tools=herramientas, prompt=prompt)

    return AgentExecutor(
        agent=agente_base,
        tools=herramientas,
        verbose=True,
        max_iterations=25,  # 🔥 Aumentado de 15 a 25 para consultas complejas
        handle_parsing_errors=True,  # 🔥 No colapsa ante errores de parsing
    )