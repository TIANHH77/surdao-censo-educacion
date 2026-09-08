import streamlit as st
import pandas as pd
import numpy as np
import os
import pydeck as pdk
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
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
        "Google: Gemini 3.5 Flash (Rápido y ligero, GA)",
        "Google: Gemini 3.1 Pro (Razonamiento profundo)",
        "Groq: Llama-3.3-70b (Equilibrado)",
        "Groq: Mixtral-8x7b",
        "Local: Ollama (Llama 3)",
    ],
)

# Instanciación del LLM
if "Gemini 3.1 Pro" in modelo_elegido:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro", temperature=0) # Actualizado al nombre de modelo estándar
elif "Gemini 3.5 Flash" in modelo_elegido:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
elif "Llama-3.3" in modelo_elegido:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
elif "Mixtral" in modelo_elegido:
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
else:
    llm = ChatOllama(model="qwen2.5-coder:14b", base_url="http://localhost:11434", temperature=0)

st.sidebar.success(f"Motor activo: {modelo_elegido}")

# Control de Costos
st.sidebar.markdown("---")
st.sidebar.subheader("💸 Control de Costos (Ahorro API)")
modo_economico = st.sidebar.checkbox(
    "Modo económico activo",
    value=True,
    help="Limita tokens y previene que el agente entre en loops infinitos.",
)
max_iter = 4 if modo_economico else 10
max_chars_salida_tool = 2000 if modo_economico else 8000

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
    """Ahorra tokens convirtiendo tablas de 300 columnas a formato largo."""
    if df.shape[1] <= UMBRAL_COLUMNAS_ANCHAS:
        return df

    id_vars = [c for c in df.columns if c in COLUMNAS_META_CANDIDATAS]
    value_vars = [c for c in df.columns if c not in id_vars]

    if not id_vars or not value_vars:
        return df

    return df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="categoria_destino",
        value_name="valor",
    )

@st.cache_data
def cargar_datos():
    base_censo = "data/CENSO_2024"
    base_edu = "data/educacion"
    
    # Manejo de error silencioso si falta un archivo (útil en dev)
    dfs = {}
    archivos = {
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

    for nombre, ruta in archivos.items():
        if os.path.exists(ruta):
            df = pd.read_parquet(ruta)
            
            # Normalización rápida
            if "nombre_comuna" in df.columns:
              df["nombre_comuna"] = (
              df["nombre_comuna"]
              .astype(str)
              .str.strip()
              .str.upper()  # o .str.title(), pero el MISMO criterio en TODAS las tablas
    
            )
            if "CUT" in df.columns:
                df = df.rename(columns={"CUT": "codigo_comuna"})
            if "codigo_comuna" in df.columns:
                df["codigo_comuna"] = pd.to_numeric(df["codigo_comuna"], errors="coerce").astype("Int64")
            if "COMUNA" in df.columns:
                df = df.rename(columns={"COMUNA": "nombre_comuna"})
            elif "comuna" in df.columns:
                df = df.rename(columns={"comuna": "nombre_comuna"})

            dfs[nombre] = aplanar_si_es_ancha(nombre, df)
            
    return dfs

diccionario_dfs = cargar_datos()
if not diccionario_dfs:
    st.error("No se encontraron los archivos parquet. Revisa la ruta 'data/'.")
    st.stop()

# ==========================================
# 4. SELECTOR DE TABLAS (Ahorro Contexto)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Tablas Activas")

archivos_seleccionados = st.sidebar.multiselect(
    "Selecciona qué tablas cruzar hoy:",
    options=list(diccionario_dfs.keys()),
    default=["Matriz_Educacion", "Censo_Edad_Envejecimiento"] if "Matriz_Educacion" in diccionario_dfs else list(diccionario_dfs.keys())[:1],
    help="Menos tablas = respuestas más rápidas y menos consumo de la API.",
)

with st.sidebar.expander("🔍 Explorar Esquema"):
    for nombre in archivos_seleccionados:
        st.markdown(f"**{nombre}** ({diccionario_dfs[nombre].shape[0]} filas)")
        st.caption(f"_{', '.join(diccionario_dfs[nombre].columns[:10])}..._")

# ==========================================
# 5. MAPA PYDECK (Omitido visualmente en código para ahorrar espacio aquí, pero déjalo tal cual lo tienes)
# ==========================================
# (Aquí va tu código original de PyDeck, está perfecto)

# ==========================================
## ==========================================
# 6. CONFIGURACIÓN DEL AGENTE Y TOOL CALLING
# ==========================================
# 6. CONFIGURACIÓN DEL AGENTE Y TOOL CALLING
# ==========================================
if not archivos_seleccionados:
    st.warning("Selecciona al menos una tabla para iniciar.")
    st.stop()
 
dfs_seleccionados = {nombre: diccionario_dfs[nombre] for nombre in archivos_seleccionados}
 
 
def construir_esquema_compacto(dfs_dict):
    partes = []
    for nombre, df in dfs_dict.items():
        cols_tipos = ", ".join(f"{c}({t})" for c, t in df.dtypes.items())
        partes.append(f"- **{nombre}**: {cols_tipos}")
    return "\n".join(partes)
 
 
# Herramienta de Python genérica (escape hatch para lo que las tools específicas no cubren)
_python_tool = PythonAstREPLTool(locals={"pd": pd, "np": np, **dfs_seleccionados})
 
 
@tool
def ejecutar_pandas(codigo: str) -> str:
    """Ejecuta código Python/Pandas en memoria.
    REGLA VITAL: Devuelve siempre data agregada.
    Usa .head(10), .describe(), o agrupaciones. NUNCA devuelvas un dataframe entero."""
    try:
        resultado = _python_tool.run(codigo)
    except Exception as e:
        return f"ERROR de Python: {e}. Revisa la sintaxis o los nombres de las columnas."
 
    texto = str(resultado)
    if len(texto) > max_chars_salida_tool:
        texto = texto[:max_chars_salida_tool] + "\n...[TRUNCADO POR AHORRO DE TOKENS]"
    return texto
 
 
@tool
def ranking_comunas_por_notas(matricula_minima: int = 500, top_n: int = 15) -> str:
    """Devuelve el ranking de comunas por promedio de notas, PONDERADO por matrícula
    (no un promedio simple de filas) y filtrando comunas con matrícula total menor a
    matricula_minima para evitar distorsión por muestras pequeñas.
    Úsala SIEMPRE que te pregunten por 'mejor/peor comuna' en rendimiento académico."""
    df = diccionario_dfs["Matriz_Educacion"]
    resumen = df.groupby("nombre_comuna").apply(
        lambda g: pd.Series(
            {
                "promedio_ponderado": (g["Promedio_Notas"] * g["Total_Alumnos"]).sum() / g["Total_Alumnos"].sum(),
                "alumnos_totales": g["Total_Alumnos"].sum(),
                "n_colegios": g["RBD"].nunique(),
            }
        ),
        include_groups=False,
    )
    resumen = resumen[resumen["alumnos_totales"] >= matricula_minima]
    resultado = resumen.sort_values("promedio_ponderado", ascending=False).head(top_n)
    return resultado.to_markdown()
 
 
@tool
def perfil_comuna(nombre_comuna: str) -> str:
    """Devuelve un perfil completo de una comuna: promedio de notas ponderado y volatilidad
    (Matriz_Educacion), años de escolaridad promedio (Censo), tasa de alfabetización (Censo)
    y nivel educativo (Censo). Usa esta tool SIEMPRE que pregunten por datos de UNA comuna
    específica cruzando educación con censo. El nombre de comuna se normaliza automáticamente."""
    comuna = nombre_comuna.strip().upper()
    perfil = {}
 
    edu = diccionario_dfs["Matriz_Educacion"]
    edu_c = edu[edu["nombre_comuna"] == comuna]
    if not edu_c.empty and edu_c["Total_Alumnos"].sum() > 0:
        perfil["promedio_notas_ponderado"] = round(
            (edu_c["Promedio_Notas"] * edu_c["Total_Alumnos"]).sum() / edu_c["Total_Alumnos"].sum(), 3
        )
        perfil["volatilidad_promedio"] = round(edu_c["Volatilidad_Rendimiento"].mean(), 3)
        perfil["n_colegios"] = edu_c["RBD"].nunique()
    else:
        perfil["educacion"] = "SIN DATOS para esta comuna en Matriz_Educacion"
 
    esc = diccionario_dfs["Censo_Escolaridad_Promedio"]
    esc_c = esc[esc["nombre_comuna"] == comuna]
    perfil["años_escolaridad_promedio"] = (
        round(esc_c["años_de_escolaridad_promedio"].mean(), 2) if not esc_c.empty else "SIN DATOS"
    )
 
    alf = diccionario_dfs["Censo_Alfabetizacion"]
    alf_c = alf[alf["nombre_comuna"] == comuna]
    if not alf_c.empty and alf_c["poblacion_de_5_años_o_más"].sum() > 0:
        perfil["tasa_alfabetizacion_pct"] = round(
            100 * alf_c["sabe_leer_y_escribir"].sum() / alf_c["poblacion_de_5_años_o_más"].sum(), 1
        )
    else:
        perfil["alfabetizacion"] = "SIN DATOS"
 
    if not any("SIN DATOS" not in str(v) for v in perfil.values()):
        return f"No se encontró la comuna '{nombre_comuna}' en ninguna tabla. Verifica el nombre exacto."
 
    return f"Perfil de {comuna}:\n" + "\n".join(f"- {k}: {v}" for k, v in perfil.items())
 
 
@tool
def listar_establecimientos_comuna(nombre_comuna: str, anio: int | None = None) -> str:
    """Lista los establecimientos (colegios) ÚNICOS de una comuna, deduplicados por RBD
    (Matriz_Educacion tiene una fila por RBD POR AÑO, así que un mismo colegio aparece
    varias veces si no se deduplica). Si no se especifica año, usa el año más reciente
    disponible para cada RBD. Úsala SIEMPRE que pidan listar/mostrar colegios de una comuna."""
    comuna = nombre_comuna.strip().upper()
    df = diccionario_dfs["Matriz_Educacion"]
    df_c = df[df["nombre_comuna"] == comuna]
 
    if df_c.empty:
        return f"No se encontraron establecimientos para la comuna '{nombre_comuna}'."
 
    if anio is not None:
        df_c = df_c[df_c["Anio"] == anio]
        if df_c.empty:
            return f"No hay datos de '{nombre_comuna}' para el año {anio}."
    else:
        # Se queda con el registro más reciente de cada RBD
        df_c = df_c.sort_values("Anio", ascending=False).drop_duplicates(subset="RBD", keep="first")
 
    columnas = [c for c in ["RBD", "Nombre_Colegio", "Total_Alumnos", "Ratio_Alumnos_Docente", "Promedio_Notas", "Anio"] if c in df_c.columns]
    resultado = df_c[columnas].sort_values("Total_Alumnos", ascending=False)
    return f"{len(resultado)} establecimiento(s) único(s) en {comuna}:\n" + resultado.to_markdown(index=False)
 
 
# Lista única de herramientas — construida DESPUÉS de definir todas las funciones @tool
tools = [
    perfil_comuna,
    ranking_comunas_por_notas,
    listar_establecimientos_comuna,
    ejecutar_pandas,  # escape hatch genérico, al final para que el LLM prefiera las específicas
]
 
prompt_sistema = f"""Eres un Analista de Datos Senior de Sur DAO.
 
TABLAS DISPONIBLES EN MEMORIA:
{construir_esquema_compacto(dfs_seleccionados)}
 
INSTRUCCIONES ESTRICTAS:
1. Antes de usar 'ejecutar_pandas', evalúa si 'perfil_comuna', 'ranking_comunas_por_notas'
   o 'listar_establecimientos_comuna' ya resuelven la pregunta. Prefiérelas siempre que apliquen,
   porque tienen la lógica de cálculo correcta ya validada (ponderación, deduplicación, etc.).
2. Para hacer un JOIN/MERGE entre tablas con 'ejecutar_pandas' usa pd.merge(..., on='codigo_comuna', how='inner').
3. Los nombres de comuna en TODAS las tablas están normalizados en MAYÚSCULAS (ej: 'LO PRADO').
   Al filtrar por comuna, usa siempre el nombre en mayúsculas.
4. Matriz_Educacion tiene UNA FILA POR RBD POR AÑO (dato panel, 2012-2024). Nunca presentes
   múltiples años del mismo RBD como si fueran colegios distintos.
5. Al rankear "mejor/peor comuna" por notas, usa SIEMPRE promedio ponderado por Total_Alumnos
   (nunca .mean() simple sobre filas), y aplica un filtro mínimo de matrícula.
6. Si el resultado de una herramienta contiene NaN, None, o está vacío, DEBES responder
   "No encontré datos para esa consulta con las tablas actuales" y explicar qué faltó.
   JAMÁS reemplaces un NaN por un número inventado, y JAMÁS reutilices un valor de una
   pregunta anterior para rellenar una respuesta distinta.
7. NUNCA reimportes pandas ni redefinas las tablas con datos de ejemplo/ficticios. Las
   variables pd, np y las tablas YA EXISTEN en memoria — úsalas directamente.
8. Responde al usuario de forma clara, directa, en español, resumiendo los hallazgos.
"""
 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_sistema),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
 
agente_base = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agente = AgentExecutor(
    agent=agente_base,
    tools=tools,
    verbose=True,
    max_iterations=max_iter,
    early_stopping_method="force",
    handle_parsing_errors=True,
)
 
# ==========================================
# 7. INTERFAZ DE CHAT Y MEMORIA NATIVA
# ==========================================
if "mensajes_ui" not in st.session_state:
    st.session_state.mensajes_ui = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Dibujar historial en la interfaz
for msg in st.session_state.mensajes_ui:
    st.chat_message(msg["role"]).write(msg["content"])

pregunta = st.chat_input("Pide cruzar datos, ej: '¿Qué comuna tiene el mejor promedio escolar y mayor alfabetización?'")

if pregunta:
    # 1. Mostrar pregunta en UI
    st.session_state.mensajes_ui.append({"role": "user", "content": pregunta})
    st.chat_message("user").write(pregunta)

    # 2. Ejecutar Agente (pasando la memoria nativa de LangChain)
    with st.spinner("🕵️‍♂️ Cruzando datos con Pandas..."):
        try:
            respuesta = agente.invoke({
                "input": pregunta,
                "chat_history": st.session_state.chat_history
            })
            output = respuesta["output"]

            # 3. Guardar en memoria de LangChain
            st.session_state.chat_history.append(HumanMessage(content=pregunta))
            st.session_state.chat_history.append(AIMessage(content=output))

            # 4. Mostrar respuesta en UI
            st.session_state.mensajes_ui.append({"role": "assistant", "content": output})
            st.chat_message("assistant").write(output)

        except Exception as e:
            st.error(f"El Agente tropezó procesando los datos: {e}")
            st.caption("Si usas APIs gratuitas, verifica no haber excedido la cuota por minuto (Rate Limit 429).")