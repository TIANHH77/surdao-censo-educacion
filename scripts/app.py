import streamlit as st
import pandas as pd
import os
import pydeck as pdk
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Sur DAO 2.0", layout="wide")

# ==========================================
# 1. CONFIGURACIÓN DE OMNIROUTE
# ==========================================
os.environ["OPENAI_API_KEY"] = "omniroute-local-key"
os.environ["OPENAI_API_BASE"] = "http://localhost:20128/v1"

st.title("🤖 Sur DAO 2.0 - Motor de Datos Híbrido")
st.markdown("Agente analítico con datamart selectivo, manual metodológico (RAG) y panel geográfico interactivo.")

# ==========================================
# 2. CARGA DEL ORÁCULO METODOLÓGICO (RAG)
# ==========================================
@st.cache_resource(show_spinner="Vectorizando el manual del Censo...")
def cargar_manual_censo():
    if not os.path.exists("data/manual_uso_microdatos_censo2024.pdf"):
        return None
    loader = PyPDFLoader("data/manual_uso_microdatos_censo2024.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_manual = cargar_manual_censo()

@tool
def consultar_manual_censo(query: str) -> str:
    """Usa esta herramienta EXCLUSIVAMENTE para buscar definiciones metodológicas y fórmulas del Censo 2024."""
    if not retriever_manual:
        return "El manual no está disponible en la carpeta data/."
    resultados = retriever_manual.invoke(query)
    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    return f"Información oficial del manual:\n{contexto}"

# ==========================================
# 3. CARGA SELECTIVA CON FIX DE DUPLICADOS
# ==========================================
@st.cache_data(show_spinner="Cargando datamart selectivo en memoria...")
def cargar_datos_seleccionados():
    dfs = {}
    archivos_mapa = {
    # Los que ya tenías
    "Histórico Educativo (2012-2023)": "educacion_historico_2012_2023.parquet",
    "Matriz 2024 + Censo": "educacion_censo_2024.parquet",
    "Alfabetización Comunal (P7_10)": "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet",
    "Nivel Educativo Comunal (P7_2)": "P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet",
    "Asistencia Neta Comunal (P7_8)": "P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet",
    
    # 🔥 LOS NUEVOS PODERES
    "Escolaridad Inmigrantes (P8_2)": "P8_2_Años_de_escolaridad_promedio_para_la_población_inmigrante_.parquet",
    "Envejecimiento (D2_2)": "D2_2_Población_censada_por_tramo_de_edad_e_índice_de_envejecimi.parquet",
    "Inmigrantes por País (D4_4)": "D4_4_Inmigrantes_internacionales_por_país_de_nacimiento_según_c.parquet",
    "Migración Interna (D5_2)": "D5_2_Población_censada_por_comuna_de_residencia_habitual_hace_5.parquet",
    "Discapacidad (P1_2)": "P1_2_Población_de_5_años_o_más_con_discapacidad_por_sexo_según_.parquet",
    "Pueblos Originarios (P2_2)": "P2_2_Población_que_es_o_se_considera_perteneciente_a_un_pueblo_.parquet",
    "Años Escolaridad (P7_4)": "P7_4_Años_de_escolaridad_promedio_según_sexo_y_comuna.parquet"
}
    
    for nombre_amigable, nombre_archivo in archivos_mapa.items():
        ruta = os.path.join("data", nombre_archivo)
        if os.path.exists(ruta):
            try:
                df = pd.read_parquet(ruta)
                df = df.drop_duplicates()  # Fix de duplicados
                dfs[nombre_amigable] = df
            except Exception as e:
                print(f"Error al cargar {nombre_archivo}: {e}")
        else:
            print(f"Advertencia: No se encontró {ruta}")
            
    return dfs

diccionario_dfs = cargar_datos_seleccionados()

st.sidebar.header("📁 Bases Activas en Memoria")
for nombre, df in diccionario_dfs.items():
    st.sidebar.success(f"**{nombre}**: {df.shape[0]:,} filas")

# ==========================================
# 4. CONFIGURACIÓN DEL AGENTE Y HERRAMIENTAS
# ==========================================
llm = ChatOpenAI(model="oc/deepseek-v4-flash-free", temperature=0)

_python_tool = PythonAstREPLTool(locals={"pd": pd, "dfs": diccionario_dfs})

@tool
def ejecutar_pandas(codigo: str) -> str:
    """Ejecuta código Python/pandas contra el diccionario de DataFrames 'dfs'."""
    try:
        resultado = _python_tool.run(codigo)
        return str(resultado)[:3000]
    except Exception as e:
        return f"ERROR al ejecutar: {e}"

herramientas = [ejecutar_pandas, consultar_manual_censo]

prompt_sistema = """Eres el Agente Principal de Sur DAO. Tienes acceso a un datamart selectivo optimizado para análisis educativo y censal.

LAS TABLAS ESTÁN DISPONIBLES EN UN DICCIONARIO LLAMADO `dfs`:
- dfs["Histórico Educativo (2012-2023)"] -> Columnas: RBD, Nombre_Colegio, Total_Alumnos, Total_Docentes, Promedio_Notas, Anio, COMUNA, REGION, LATITUD, LONGITUD, Volatilidad_Rendimiento
- dfs["Matriz 2024 + Censo"] -> Columnas: RBD, Nombre_Colegio, Total_Alumnos, Total_Docentes, Promedio_Notas, comuna, LATITUD, LONGITUD, sabe_leer_y_escribir, parvularia, básica, media, superior

REGLAS OBLIGATORIAS:
1. Usa siempre la notación: `df = dfs["Nombre exacto de la tabla"]`.
2. FILTRADO ESTRICTO: Para comunas, busca el nombre completo exacto ignorando mayúsculas (ej: `df[df['comuna'].str.lower() == 'isla de maipo']`).
3. NUNCA inventes datos. Usa 'ejecutar_pandas' para responder con precisión matemática.
4. Cuando la pregunta requiera varios cálculos relacionados (ej: notas + volatilidad + ratio docente), 
5. USO DEL MANUAL: Consulta el manual metodológico SOLO si el usuario te pide explícitamente definir una variable o entender un concepto. Para todo lo que sea extraer datos o cruzar información, confía únicamente en tu herramienta 'ejecutar_pandas'.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_sistema),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agente_base = create_tool_calling_agent(llm=llm, tools=herramientas, prompt=prompt)
agente = AgentExecutor(agent=agente_base, tools=herramientas, verbose=True, max_iterations=15)

# ==========================================
# 5. GESTIÓN DE ESTADO PARA EL MAPA PERSISTENTE (CON PYDECK)
# ==========================================
if "df_mapa" not in st.session_state:
    st.session_state.df_mapa = None
if "titulo_mapa" not in st.session_state:
    st.session_state.titulo_mapa = ""

# Si hay un mapa activo en la memoria, lo renderizamos con PyDeck
if st.session_state.df_mapa is not None:
    with st.container():
        st.subheader(st.session_state.titulo_mapa)

        df_mapa = st.session_state.df_mapa

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa,
            get_position=["lon", "lat"],
            get_radius=100,
            get_fill_color=[41, 128, 185, 200], # Color azul corporativo
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=df_mapa["lat"].mean(),
            longitude=df_mapa["lon"].mean(),
            zoom=13,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>Colegio:</b> {Nombre_Colegio}",
                    "style": {"background": "grey", "color": "white", "padding": "5px"},
                },
                map_style="light",
            )
        )

        if st.button("🗑️ Cerrar / Ocultar Mapa"):
            st.session_state.df_mapa = None
            st.rerun()
    st.markdown("---")

# ==========================================
# 6. CHAT UI & DETECCIÓN AUTOMÁTICA DE MAPAS
# ==========================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    st.chat_message(msg["role"]).write(msg["content"])

pregunta = st.chat_input("Pregúntale al Agente (ej: 'Muéstrame el mapa de los colegios de Isla de Maipo')")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.chat_message("user").write(pregunta)

    chat_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.mensajes[:-1]]

    with st.spinner("Procesando consulta..."):
        try:
            respuesta = agente.invoke({"input": pregunta, "chat_history": chat_history})
            output_final = respuesta["output"]

            # 1. Verificamos si se pasó de iteraciones ANTES de guardarlo
            if "stopped due to max iterations" in output_final.lower() or "agent stopped" in output_final.lower():
                output_final = (
                    "¡Excelente pregunta! 🙌 De hecho es tan compleja que superó mi límite de pasos "
                    "de análisis para responderla de una sola vez.\n\n"
                    "¿Podrías reformularla dividiéndola en partes más acotadas? Por ejemplo, en vez de "
                    "pedir todo junto (mejores notas + volatilidad + varias comunas a la vez), probá primero "
                    "con una sola comuna o una sola métrica, y después seguimos con el resto. 😊"
                )

            # 2. Guardamos y mostramos la respuesta definitiva (sea la original o la amable)
            st.session_state.mensajes.append({"role": "assistant", "content": output_final})
            st.chat_message("assistant").write(output_final)

            # --- ACTUALIZACIÓN DINÁMICA E INTELIGENTE DEL MAPA PERSISTENTE ---
            texto_lower = pregunta.lower()

            df_m = diccionario_dfs["Matriz 2024 + Censo"]
            comunas_disponibles = df_m['comuna'].dropna().unique()

            comuna_detectada = None
            for c in comunas_disponibles:
                if c.strip().lower() in texto_lower:
                    comuna_detectada = c
                    break

            # Si mencionó "mapa" pero no reconocimos ninguna comuna nueva en el texto,
            # reusamos la última comuna que ya estaba mostrando (si había alguna)
            if comuna_detectada is None and "mapa" in texto_lower and "titulo_mapa" in st.session_state and st.session_state.titulo_mapa:
                comuna_detectada = st.session_state.titulo_mapa.split(" en ")[-1]

            if comuna_detectada:
                df_comuna = df_m[df_m['comuna'].str.strip().str.lower() == comuna_detectada.lower()].drop_duplicates(subset=['RBD'])
                
                if 'LATITUD' in df_comuna.columns and 'LONGITUD' in df_comuna.columns:
                    df_mapa = df_comuna[['LATITUD', 'LONGITUD', 'Nombre_Colegio']].dropna()
                    if not df_mapa.empty:
                        df_mapa = df_mapa.rename(columns={'LATITUD': 'lat', 'LONGITUD': 'lon'})
                        st.session_state.df_mapa = df_mapa
                        st.session_state.titulo_mapa = f"🗺️ Ubicación de Establecimientos en {comuna_detectada.title()}"
                        st.rerun()  # Recargamos para que el mapa aparezca fijo arriba de inmediato

        except Exception as e:
            st.error("Error detallado en el agente:")
            st.exception(e)