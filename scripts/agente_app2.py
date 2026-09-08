import streamlit as st
import pandas as pd
import os
import pydeck as pdk
from dotenv import load_dotenv

# LangChain Imports actualizados
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
load_dotenv()

st.set_page_config(page_title="Auditor de Datos - Sur DAO", page_icon="🤖", layout="wide")
st.title("🤖 Agente Auditor Sur DAO")
st.markdown("Analiza y cruza datos del rendimiento escolar histórico (2012-2024) con el Censo 2024.")

# ==========================================
# 2. SELECTOR DE MODELOS & COSTOS
# ==========================================
st.sidebar.header("⚙️ Configuración del Motor")
modelo_elegido = st.sidebar.selectbox(
    "Selecciona el modelo de IA:",
    [
        "Google: Gemini 1.5 Pro (Razonamiento profundo)",
        "Google: Gemini 1.5 Flash (Rápido y ligero)",
        "Groq: Llama-3.3-70b (Equilibrado)",
        "Groq: Mixtral-8x7b",
        "Local: Ollama (Llama 3)",
    ],
)

if "Gemini 1.5 Pro" in modelo_elegido:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
elif "Gemini 1.5 Flash" in modelo_elegido:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
elif "Llama-3.3" in modelo_elegido:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
elif "Mixtral" in modelo_elegido:
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
else:
    llm = ChatOllama(model="llama3", base_url="http://host.docker.internal:11434", temperature=0)

st.sidebar.success(f"Motor activo: {modelo_elegido}")

st.sidebar.markdown("---")
st.sidebar.subheader("💸 Control de Costos")
modo_economico = st.sidebar.checkbox(
    "Modo económico (recomendado)",
    value=True,
    help="Limita las iteraciones del agente y el tamaño de las respuestas intermedias para no gastar tokens de más.",
)
max_iter = 4 if modo_economico else 10
max_chars_salida_tool = 2500 if modo_economico else 8000

# ==========================================
# 3. CARGA Y NORMALIZACIÓN DE DATOS
# ==========================================
UMBRAL_COLUMNAS_ANCHAS = 40
COLUMNAS_META_CANDIDATAS = {
    "codigo_region", "region", "codigo_provincia", "provincia",
    "codigo_comuna", "comuna", "sexo", "poblacion_censada",
    "no_migrante_interno_regional", "no_migrante_interno_comunal",
    "region_de_residencia_habitual_actual",
    "comuna_de_residencia_habitual_actual",
    "provincia_de_residencia_habitual_actual",
    "aún_no_nacian_(menores_de_5_años)",
}

def aplanar_si_es_ancha(nombre: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tablas 'anchas' (1 columna por comuna/región) a formato largo."""
    if df.shape[1] <= UMBRAL_COLUMNAS_ANCHAS:
        return df

    id_vars = [c for c in df.columns if c in COLUMNAS_META_CANDIDATAS]
    value_vars = [c for c in df.columns if c not in id_vars]

    if not id_vars or not value_vars:
        st.sidebar.warning(f"⚠️ '{nombre}' parece ancha ({df.shape[1]} cols) pero no se aplanó automáticamente.")
        return df

    df_largo = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="categoria_destino",
        value_name="valor",
    )
    return df_largo


@st.cache_data
def cargar_datos():
    base_censo = "data/CENSO_2024"
    base_edu = "data/educacion"

    dfs = {}
    archivos_esperados = {
        "Matriz_Educacion": os.path.join(base_edu, "dataset_auditoria_final.parquet"),
        "Censo_Edad_Envejecimiento": os.path.join(base_censo, "D2_2_Población_censada_por_tramo_de_edad_e_índice_de_envejecimi.parquet"),
        "Censo_Escolaridad_Inmigrantes": os.path.join(base_censo, "P8_2_Años_de_escolaridad_promedio_para_la_población_inmigrante_.parquet"),
        "Censo_Alfabetizacion": os.path.join(base_censo, "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet"),
        "Censo_Asistencia_Neta": os.path.join(base_censo, "P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet"),
        "Censo_Escolaridad_Promedio": os.path.join(base_censo, "P7_4_Años_de_escolaridad_promedio_según_sexo_y_comuna.parquet"),
        "Censo_Nivel_Educativo": os.path.join(base_censo, "P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet"),
        "Censo_Pueblos_Originarios": os.path.join(base_censo, "P2_2_Población_que_es_o_se_considera_perteneciente_a_un_pueblo_.parquet"),
        "Censo_Discapacidad": os.path.join(base_censo, "P1_2_Población_de_5_años_o_más_con_discapacidad_por_sexo_según_.parquet"),
        "Censo_Migracion_Interna": os.path.join(base_censo, "D5_2_Población_censada_por_comuna_de_residencia_habitual_hace_5.parquet"),
        "Censo_Inmigracion_Internac": os.path.join(base_censo, "D4_4_Inmigrantes_internacionales_por_país_de_nacimiento_según_c.parquet"),
    }

    for nombre, ruta in archivos_esperados.items():
        if os.path.exists(ruta):
            df = pd.read_parquet(ruta)
            if "CUT" in df.columns:
                df = df.rename(columns={"CUT": "codigo_comuna"})
            if "codigo_comuna" in df.columns:
                df["codigo_comuna"] = pd.to_numeric(df["codigo_comuna"], errors="coerce").astype("Int64")
            if "COMUNA" in df.columns:
                df = df.rename(columns={"COMUNA": "nombre_comuna"})
            elif "comuna" in df.columns:
                df = df.rename(columns={"comuna": "nombre_comuna"})

            df = aplanar_si_es_ancha(nombre, df)
            dfs[nombre] = df
        else:
            st.sidebar.warning(f"⚠️ Archivo no encontrado: {nombre}")

    return dfs


try:
    diccionario_dfs = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar las tablas. Verifica que la carpeta 'data' exista: {e}")
    st.stop()

# ==========================================
# 4. SELECTOR MÚLTIPLE DE DATASETS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Gestión de Bases de Datos")

nombres_archivos = list(diccionario_dfs.keys())
archivos_seleccionados = st.sidebar.multiselect(
    "Selecciona qué tablas cruzar (¡salva tokens!):",
    options=nombres_archivos,
    default=["Matriz_Educacion", "Censo_Edad_Envejecimiento"] if "Matriz_Educacion" in nombres_archivos else nombres_archivos[:2],
    help="Selecciona máx. 2 o 3 tablas por consulta. Menos tablas = menos tokens = menos costo.",
)

with st.sidebar.expander("🔍 Ver columnas de las tablas elegidas"):
    if not archivos_seleccionados:
        st.warning("Selecciona al menos una tabla arriba.")
    else:
        for nombre in archivos_seleccionados:
            df_tmp = diccionario_dfs[nombre]
            st.markdown(f"**{nombre}** — {df_tmp.shape[0]} filas x {df_tmp.shape[1]} cols")
            st.caption(f"_{', '.join(df_tmp.columns.tolist())}_")

# ==========================================
# 5. MAPA INTERACTIVO (PYDECK 3D/2D)
# ==========================================
if "Matriz_Educacion" in archivos_seleccionados and "Matriz_Educacion" in diccionario_dfs:
    st.subheader("📍 Vista Geoespacial de Colegios")
    df_educacion = diccionario_dfs["Matriz_Educacion"]
    columna_anio = "Anio"

    col1, col2 = st.columns(2)

    with col1:
        if columna_anio in df_educacion.columns:
            anios_disponibles = sorted(df_educacion[columna_anio].dropna().unique())
            anio_seleccionado = st.select_slider("⏳ Selecciona el Año:", options=anios_disponibles, value=max(anios_disponibles))
            df_temporal = df_educacion[df_educacion[columna_anio] == anio_seleccionado]
        else:
            st.warning(f"No se encontró la columna '{columna_anio}'.")
            df_temporal = df_educacion

    with col2:
        if "nombre_comuna" in df_temporal.columns:
            comunas_disponibles = df_temporal["nombre_comuna"].dropna().unique()
            comuna_mapa = st.selectbox("🗺️ Filtra por Comuna:", ["Todas"] + sorted(list(comunas_disponibles)))
            df_mapa = df_temporal[df_temporal["nombre_comuna"] == comuna_mapa] if comuna_mapa != "Todas" else df_temporal
        else:
            df_mapa = df_temporal

    if "LATITUD" in df_mapa.columns and "LONGITUD" in df_mapa.columns:
        lat_centro = df_mapa["LATITUD"].mean()
        lon_centro = df_mapa["LONGITUD"].mean()
        zoom_inicial = 11 if comuna_mapa != "Todas" else 4

        tema_elegido = st.radio("🎨 Estilo del Mapa:", ["🌙 Oscuro", "☀️ Claro (Light)"], horizontal=True)
        estilo_pydeck = "light" if "Claro" in tema_elegido else "dark"

        df_mapa_limpio = df_mapa.copy()
        col_promedio = "Promedio_Notas_Anio" if "Promedio_Notas_Anio" in df_mapa_limpio.columns else "Promedio_Notas"
        col_volatilidad = "Volatilidad_Rendimiento"
        col_ratio = "Ratio_Alumnos_Docente"

        for col in [col_promedio, col_volatilidad, col_ratio, "Total_Alumnos", "Total_Docentes"]:
            if col in df_mapa_limpio.columns:
                df_mapa_limpio[col] = df_mapa_limpio[col].fillna("Sin registro")

        capa_colegios = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa_limpio,
            get_position="[LONGITUD, LATITUD]",
            get_color="[220, 50, 50, 200]",
            get_radius=250,
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style=estilo_pydeck,
                initial_view_state=pdk.ViewState(latitude=lat_centro, longitude=lon_centro, zoom=zoom_inicial, pitch=0),
                layers=[capa_colegios],
                tooltip={
                    "html": "<b>🏫 {Nombre_Colegio}</b><br/>"
                    f"📚 Promedio: <b>{{{col_promedio}}}</b><br/>"
                    f"📉 Volatilidad: <b>{{{col_volatilidad}}}</b><br/>"
                    f"⚖️ Ratio Alumnos/Profe: <b>{{{col_ratio}}}</b><br/>"
                    "👥 Alumnos: {Total_Alumnos} | 👨‍🏫 Docentes: {Total_Docentes}",
                    "style": {"backgroundColor": "#222222", "color": "white", "borderRadius": "8px", "padding": "12px"}
                },
            )
        )
    else:
        st.warning("El dataset no tiene las coordenadas (LATITUD/LONGITUD) necesarias para el mapa.")

    st.markdown("---")

# ==========================================
# 6. AGENTE LIVIANO (Tool-calling)
# ==========================================
if not archivos_seleccionados:
    st.info("👈 Por favor, selecciona al menos una base de datos en el menú lateral para iniciar el chat.")
    st.stop()

dfs_seleccionados = {nombre: diccionario_dfs[nombre] for nombre in archivos_seleccionados}

def construir_esquema_compacto(dfs_dict: dict) -> str:
    partes = []
    for nombre, df in dfs_dict.items():
        cols_tipos = ", ".join(f"{c}({str(t)})" for c, t in df.dtypes.items())
        partes.append(f"- **{nombre}** ({df.shape[0]} filas): {cols_tipos}")
    return "\n".join(partes)

esquema_texto = construir_esquema_compacto(dfs_seleccionados)

_python_tool = PythonAstREPLTool(locals={"pd": pd, **dfs_seleccionados})

@tool
def ejecutar_pandas(codigo: str) -> str:
    """Ejecuta código Python/pandas contra los DataFrames ya cargados."""
    try:
        resultado = _python_tool.run(codigo)
    except Exception as e:
        return f"ERROR al ejecutar el código: {e}"

    texto = str(resultado)
    if len(texto) > max_chars_salida_tool:
        texto = texto[:max_chars_salida_tool] + f"\n... [resultado truncado a {max_chars_salida_tool} caracteres para ahorrar tokens]"
    return texto

prompt_sistema = f"""Eres un Auditor Educativo de Élite y Científico de Datos Senior.

Tienes acceso a estas tablas de pandas ya cargadas en memoria (NO las repitas ni las reimprimas completas):
{esquema_texto}

REGLAS:
1. La llave para cruzar (MERGE) entre cualquier tabla es SIEMPRE la columna: 'codigo_comuna'.
2. El nombre en texto de la comuna es SIEMPRE la columna: 'nombre_comuna' (o 'comuna' si no fue normalizada).
3. Usa la herramienta 'ejecutar_pandas' para CUALQUIER cálculo, filtro o cruce. Nunca inventes números.
4. Sé eficiente: intenta resolver la consulta en el MENOR número de llamadas a la herramienta posible.
5. Cuando ejecutes código, imprime o retorna SOLO el resultado agregado/resumido que necesitas responder.
6. Tu última respuesta DEBE empezar con la frase "Final Answer: " y estar en español, clara y concisa.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_sistema),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Aquí está la corrección clave usando create_openai_tools_agent en lugar de create_tool_calling_agent
agente_base = create_openai_tools_agent(llm=llm, tools=[ejecutar_pandas], prompt=prompt)
agente = AgentExecutor(
    agent=agente_base,
    tools=[ejecutar_pandas],
    verbose=True,
    max_iterations=max_iter,
    early_stopping_method="force",
    handle_parsing_errors=True,
)

# ==========================================
# 7. INTERFAZ DE CHAT Y MEMORIA NATIVA
# ==========================================
col_chat, col_clear = st.columns([0.85, 0.15])
with col_clear:
    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    st.chat_message(mensaje["role"]).write(mensaje["content"])

pregunta = st.chat_input("Ej: Considerando los datos cargados, ¿cuál es el promedio de notas en Santiago?")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.chat_message("user").write(pregunta)

    chat_history = []
    for msg in st.session_state.mensajes[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    with st.spinner("🕵️‍♂️ El Auditor está analizando los datos..."):
        try:
            respuesta = agente.invoke({
                "input": pregunta, 
                "chat_history": chat_history
            })
            
            output = respuesta["output"].replace("Final Answer: ", "").strip()

            st.session_state.mensajes.append({"role": "assistant", "content": output})
            st.chat_message("assistant").write(output)
            
        except Exception as e:
            st.error(f"El Agente tuvo un error procesando los datos: {e}")