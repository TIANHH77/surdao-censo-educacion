import streamlit as st
import pandas as pd
import os
import pydeck as pdk
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
load_dotenv()

st.set_page_config(page_title="Auditor de Datos - Sur DAO", page_icon="🤖", layout="wide")
st.title("🤖 Agente Auditor Sur DAO")
st.markdown("Analiza y cruza datos del rendimiento escolar histórico (2012-2024) con el Censo 2024.")

# ==========================================
# 2. SELECTOR DE MODELOS
# ==========================================
st.sidebar.header("⚙️ Configuración del Motor")
modelo_elegido = st.sidebar.selectbox(
    "Selecciona el modelo de IA:",
    [
        "Google: Gemini 1.5 Pro (Razonamiento profundo)",
        "Google: Gemini 1.5 Flash (Rápido y ligero)",
        "Groq: Llama-3.3-70b (Equilibrado)", 
        "Groq: Mixtral-8x7b", 
        "Local: Ollama (Llama 3)"
    ]
)

# Inicializar el LLM según la selección (¡Gemini Flash arreglado!)
# Inicializar el LLM según la selección - ACTUALIZADO 2026
if "Gemini 1.5 Pro" in modelo_elegido:
    # Usando el modelo de razonamiento pesado más reciente
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0)
elif "Gemini 1.5 Flash" in modelo_elegido:
    # Usando el modelo Flash más moderno y rápido
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
elif "Llama-3.3" in modelo_elegido:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
elif "Mixtral" in modelo_elegido:
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
else:
    llm = ChatOllama(model="llama3", base_url="http://host.docker.internal:11434", temperature=0)

st.sidebar.success(f"Motor activo: {modelo_elegido}")

# ==========================================
# 3. CARGA Y NORMALIZACIÓN DE DATOS 
# ==========================================
@st.cache_data
def cargar_datos():
    base_censo = "data/CENSO_2024"
    base_edu = "data/educacion"
    
    dfs = {
        "Matriz_Educacion": pd.read_parquet(os.path.join(base_edu, "dataset_auditoria_final.parquet")),
        "Censo_Edad_Envejecimiento": pd.read_parquet(os.path.join(base_censo, "D2_2_Población_censada_por_tramo_de_edad_e_índice_de_envejecimi.parquet")),
        "Censo_Escolaridad_Inmigrantes": pd.read_parquet(os.path.join(base_censo, "P8_2_Años_de_escolaridad_promedio_para_la_población_inmigrante_.parquet")),
        "Censo_Alfabetizacion": pd.read_parquet(os.path.join(base_censo, "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet")),
        "Censo_Asistencia_Neta": pd.read_parquet(os.path.join(base_censo, "P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet")),
        "Censo_Escolaridad_Promedio": pd.read_parquet(os.path.join(base_censo, "P7_4_Años_de_escolaridad_promedio_según_sexo_y_comuna.parquet")),
        "Censo_Nivel_Educativo": pd.read_parquet(os.path.join(base_censo, "P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet")),
        "Censo_Pueblos_Originarios": pd.read_parquet(os.path.join(base_censo, "P2_2_Población_que_es_o_se_considera_perteneciente_a_un_pueblo_.parquet")),
        "Censo_Discapacidad": pd.read_parquet(os.path.join(base_censo, "P1_2_Población_de_5_años_o_más_con_discapacidad_por_sexo_según_.parquet")),
        "Censo_Migracion_Interna": pd.read_parquet(os.path.join(base_censo, "D5_2_Población_censada_por_comuna_de_residencia_habitual_hace_5.parquet")),
        "Censo_Inmigracion_Internac": pd.read_parquet(os.path.join(base_censo, "D4_4_Inmigrantes_internacionales_por_país_de_nacimiento_según_c.parquet"))
    }
    
    # NORMALIZACIÓN UNIVERSAL
    for nombre, df in dfs.items():
        if 'CUT' in df.columns:
            df = df.rename(columns={'CUT': 'codigo_comuna'})
            
        if 'codigo_comuna' in df.columns:
            df['codigo_comuna'] = pd.to_numeric(df['codigo_comuna'], errors='coerce').astype('Int64')
            
        if 'COMUNA' in df.columns:
            df = df.rename(columns={'COMUNA': 'nombre_comuna'})
        elif 'comuna' in df.columns:
            df = df.rename(columns={'comuna': 'nombre_comuna'})
            
        dfs[nombre] = df
        
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
    default=["Matriz_Educacion", "Censo_Edad_Envejecimiento"], 
    help="Si usas Groq o modelos locales, selecciona máx 2 o 3 para no saturar la memoria."
)

with st.sidebar.expander("🔍 Ver columnas de las tablas elegidas"):
    if not archivos_seleccionados:
        st.warning("Selecciona al menos una tabla arriba.")
    else:
        for nombre in archivos_seleccionados:
            st.markdown(f"**{nombre}**")
            cols = ", ".join(diccionario_dfs[nombre].columns.tolist())
            st.caption(f"_{cols}_")

# ==========================================
# 5. MAPA INTERACTIVO (PYDECK 3D/2D)
# ==========================================
if "Matriz_Educacion" in archivos_seleccionados:
    st.subheader("📍 Vista Geoespacial de Colegios")
    df_educacion = diccionario_dfs["Matriz_Educacion"]
    
    # --- FILTROS VISUALES (AÑO Y COMUNA) ---
    columna_anio = 'Anio' # Asegúrate de que así se llama tu columna de año
    
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
        comunas_disponibles = df_temporal['nombre_comuna'].dropna().unique()
        comuna_mapa = st.selectbox("🗺️ Filtra por Comuna:", ["Todas"] + sorted(list(comunas_disponibles)))
        
        if comuna_mapa != "Todas":
            df_mapa = df_temporal[df_temporal['nombre_comuna'] == comuna_mapa]
        else:
            df_mapa = df_temporal

    # --- RENDERIZADO DEL MAPA CON TEMA LIGHT/DARK ---
    if 'LATITUD' in df_mapa.columns and 'LONGITUD' in df_mapa.columns:
        lat_centro = df_mapa['LATITUD'].mean()
        lon_centro = df_mapa['LONGITUD'].mean()
        zoom_inicial = 9 if comuna_mapa != "Todas" else 4

        tema_elegido = st.radio("🎨 Estilo del Mapa:", ["🌙 Oscuro", "☀️ Claro (Light)"], horizontal=True)
        estilo_pydeck = "light" if "Claro" in tema_elegido else "dark"

        # Copia para rellenar vacíos y no romper el tooltip
        df_mapa_limpio = df_mapa.copy()
        
        # Nombres exactos de las columnas
        col_promedio = 'Promedio_Notas_Anio' 
        col_volatilidad = 'Volatilidad_Rendimiento'
        col_ratio = 'Ratio_Alumnos_Docente'
        
        for col in [col_promedio, col_volatilidad, col_ratio, 'Total_Alumnos', 'Total_Docentes']:
            if col in df_mapa_limpio.columns:
                df_mapa_limpio[col] = df_mapa_limpio[col].fillna("Sin registro")

        capa_colegios = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa_limpio,
            get_position='[LONGITUD, LATITUD]',
            get_color='[220, 50, 50, 200]', 
            get_radius=250, 
            pickable=True   
        )

        st.pydeck_chart(pdk.Deck(
            map_style=estilo_pydeck,  
            initial_view_state=pdk.ViewState(
                latitude=lat_centro,
                longitude=lon_centro,
                zoom=zoom_inicial,
                pitch=0,
            ),
            layers=[capa_colegios],
            tooltip={
                "html": "<b>🏫 {Nombre_Colegio}</b><br/>"
                        f"📚 Promedio: <b>{{{col_promedio}}}</b><br/>"
                        f"📉 Volatilidad: <b>{{{col_volatilidad}}}</b><br/>"
                        f"⚖️ Ratio Alumnos/Profe: <b>{{{col_ratio}}}</b><br/>"
                        "👥 Alumnos: {Total_Alumnos} | 👨‍🏫 Docentes: {Total_Docentes}",
                "style": {
                    "backgroundColor": "#222222",
                    "color": "white",
                    "font-family": "sans-serif",
                    "border-radius": "8px",
                    "padding": "12px",
                    "max-width": "350px",         
                    "white-space": "normal",      
                    "z-index": "10000"            
                }
            }
        ))
    else:
        st.warning("El dataset no tiene las coordenadas (LATITUD/LONGITUD) necesarias para el mapa.")
    
    st.markdown("---")

# ==========================================
# 6. CONFIGURACIÓN DEL AGENTE Y CHAT
# ==========================================
dfs_para_el_agente = [diccionario_dfs[nombre] for nombre in archivos_seleccionados]

if not dfs_para_el_agente:
    st.info("👈 Por favor, selecciona al menos una base de datos en el menú lateral para iniciar el chat.")
    st.stop()

prompt_sistema = """Eres un Auditor Educativo de Élite y Científico de Datos Senior.
Tienes acceso a uno o varios DataFrames de pandas.
- IMPORTANTE: Todos los DataFrames han sido NORMALIZADOS. 
- La llave para cruzar (MERGE) entre cualquier tabla es SIEMPRE la columna: 'codigo_comuna'.
- El nombre en texto de la ciudad/comuna es SIEMPRE la columna: 'nombre_comuna'.

REGLAS:
1. CONSULTAS INDIVIDUALES: Si preguntan por un dato específico, filtra el DataFrame correspondiente y responde.
2. CRUCES (MERGE): Si te piden relacionar datos, haz pd.merge(dfX, dfY, on='codigo_comuna').
3. EXPLORACIÓN: Usa df.columns si necesitas buscar qué variables exactas existen.
4. FORMATO: Tu última respuesta DEBE empezar obligatoriamente con la frase "Final Answer: ".
"""

agente = create_pandas_dataframe_agent(
    llm=llm,
    df=dfs_para_el_agente,
    verbose=True,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    prefix=prompt_sistema
)

# Historial del Chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    st.chat_message(mensaje["role"]).write(mensaje["content"])

pregunta = st.chat_input("Ej: Considerando los datos cargados, cruza la matriz de educación con el censo...")

if pregunta:
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.chat_message("user").write(pregunta)
    
    historial_reciente = st.session_state.mensajes[-5:-1] 
    contexto_str = "Historial de la conversación reciente:\n"
    for msg in historial_reciente:
        rol = "Humano" if msg["role"] == "user" else "IA"
        contexto_str += f"- {rol}: {msg['content']}\n"
    
    pregunta_con_memoria = f"{contexto_str}\nTeniendo en cuenta el historial anterior, responde a esta petición: {pregunta}"
    
    with st.spinner("🕵️‍♂️ El Auditor está analizando, cruzando datos y haciendo cálculos..."):
        try:
            respuesta = agente.invoke(pregunta_con_memoria)
            output = respuesta["output"]
            
            st.session_state.mensajes.append({"role": "assistant", "content": output})
            st.chat_message("assistant").write(output)
        except Exception as e:
            st.error(f"El Agente tuvo un error procesando los datos: {e}")