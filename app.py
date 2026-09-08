import streamlit as st
import pandas as pd
import os
import pydeck as pdk
from langchain_core.messages import HumanMessage, AIMessage

# 🔥 Importamos EL CEREBRO desde agent_core.py
from agent_core import create_surdao_agent, invocar_con_fallback

st.set_page_config(page_title="Sur DAO 2.0", layout="wide")

col1, col2 = st.columns([1, 15])

with col1:
    try:
        st.image("assets/surdao.svg", width=60)
    except Exception:
        st.title("🏔️")

with col2:
    st.title("Sur DAO 2.0 - Motor de Datos Híbrido")

st.markdown("Agente analítico con datamart consolidado (Bases Maestras Censo 2024 + Auditoría MINEDUC), RAG y panel geográfico.")

# ==========================================
# CARGA DINÁMICA DEL DATAMART (Con soporte para subcarpetas)
# ==========================================
@st.cache_data(show_spinner="Cargando datamart consolidado en memoria...")
def cargar_datamart_completo():
    dfs = {}
    ruta_carpeta = "data"

    if not os.path.exists(ruta_carpeta):
        st.error(f"❌ ¡Alerta! La carpeta '{ruta_carpeta}' no existe. Revisa la ruta.")
        return dfs

    nombres_especiales = {
        "dataset_auditoria_final.parquet": "Auditoría Final"
    }

    # os.walk recorre automáticamente CENSO_2024, educacion y cualquier subcarpeta
    for root, dirs, files in os.walk(ruta_carpeta):
        for archivo in files:
            if archivo.endswith('.parquet'):
                ruta_completa = os.path.join(root, archivo)
                
                if archivo in nombres_especiales:
                    nombre_tabla = nombres_especiales[archivo]
                else:
                    nombre_tabla = archivo.replace('.parquet', '').replace('_', ' ')

                try:
                    df = pd.read_parquet(ruta_completa)
                    df = df.drop_duplicates()
                    dfs[nombre_tabla] = df
                except Exception as e:
                    print(f"⚠️ Error cargando {archivo}: {e}")

    return dfs

diccionario_dfs = cargar_datamart_completo()

# ==========================================
# INICIALIZACIÓN DEL AGENTE (Ignorando hasheo lento de dfs)
# ==========================================
@st.cache_resource
def inicializar_cerebro_agente(_dfs):
    print("🧠 Inicializando agente y RAG por única vez...")
    return create_surdao_agent(_dfs)

agente = inicializar_cerebro_agente(diccionario_dfs)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def generar_subpreguntas(pregunta_original: str) -> list:
    sub = []
    temas = {
        "población": ["edad", "envejecimiento", "habitantes", "censada", "cuántos", "poblacion"],
        "educación": ["escolar", "colegio", "alumno", "docente", "nota", "educacion", "escuela"],
        "migración": ["inmigrante", "extranjero", "país", "origen", "migrante", "extranjera"],
        "cuidado": ["discapacidad", "dependiente", "cuidado", "vulnerable", "adulto mayor"],
        "pueblos": ["indígena", "originario", "mapuche", "aymara", "quechua", "pueblo"],
    }
    for tema, keywords in temas.items():
        if any(k in pregunta_original.lower() for k in keywords):
            sub.append(f"📊 ¿Cuál es la **{tema}** de la comuna?")
    if not sub:
        sub = ["📊 ¿Cuánta población tiene?", "📊 ¿Cuál es el nivel educativo?", "📊 ¿Hay datos de inmigración?"]
    return sub[:3]

def detectar_comuna_en_texto(texto: str) -> str | None:
    """Busca nombres de comuna dentro del texto usando la super-tabla de auditoría."""
    if "Auditoría Final" not in diccionario_dfs:
        return None
    
    df_m = diccionario_dfs["Auditoría Final"]
    if 'comuna' in df_m.columns:
        comunas = df_m['comuna'].dropna().unique()
        for c in comunas:
            nombre_limpio = str(c).strip().lower()
            if nombre_limpio in texto.lower():
                return c
    return None

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📁 Bases Activas en Memoria")
st.sidebar.info(f"Total cargadas: {len(diccionario_dfs)} bases maestras")

busqueda = st.sidebar.text_input("🔍 Buscar tabla por nombre:", placeholder="ej: envejecimiento")

if busqueda:
    st.sidebar.subheader("Resultados:")
    for nombre, df in diccionario_dfs.items():
        if busqueda.lower() in nombre.lower():
            st.sidebar.success(f"**{nombre[:50]}** — {df.shape[0]:,} filas, {df.shape[1]} cols")
else:
    for nombre, df in diccionario_dfs.items():
        st.sidebar.success(f"**{nombre[:30]}...**: {df.shape[0]:,} filas")

# ==========================================
# GESTIÓN DE ESTADO
# ==========================================
if "df_mapa" not in st.session_state:
    st.session_state.df_mapa = None
if "titulo_mapa" not in st.session_state:
    st.session_state.titulo_mapa = ""
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "ultima_pregunta_compleja" not in st.session_state:
    st.session_state.ultima_pregunta_compleja = ""
if "sugerencias_mostradas" not in st.session_state:
    st.session_state.sugerencias_mostradas = False

# ==========================================
# MAPA PYDECK
# ==========================================
if st.session_state.df_mapa is not None:
    with st.container():
        st.subheader(st.session_state.titulo_mapa)
        df_mapa = st.session_state.df_mapa

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa,
            get_position=["lon", "lat"],
            get_radius=100,
            get_fill_color=[41, 128, 185, 200],
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

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ Cerrar Mapa"):
                st.session_state.df_mapa = None
                st.rerun()
    st.markdown("---")

# ==========================================
# CHAT UI Y LÓGICA DE AGENTE
# ==========================================
if st.session_state.sugerencias_mostradas and st.session_state.ultima_pregunta_compleja:
    st.info("💡 **Consulta muy compleja.** Prueba con una de estas subpreguntas:")
    cols = st.columns(3)
    subpreguntas = generar_subpreguntas(st.session_state.ultima_pregunta_compleja)
    for i, sub in enumerate(subpreguntas):
        with cols[i]:
            if st.button(sub, key=f"sub_{i}"):
                st.session_state.sugerencias_mostradas = False
                st.session_state.ultima_pregunta_compleja = ""
                st.session_state.mensajes.append({"role": "user", "content": sub})
                st.rerun()

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pregunta = st.chat_input("Pregúntale al Agente (ej: 'Datos de Talagante')")

if pregunta:
    st.session_state.sugerencias_mostradas = False
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Limite del historial a los últimos 20 mensajes para evitar sobrecarga de tokens
    if len(st.session_state.mensajes) > 20:
        st.session_state.mensajes = st.session_state.mensajes[-20:]

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.mensajes[:-1]
    ]

    with st.spinner("🧠 Procesando consulta..."):
        try:
            respuesta = invocar_con_fallback(agente, {"input": pregunta, "chat_history": chat_history})
            output_final = respuesta["output"]

            es_compleja = "stopped due to max iterations" in output_final.lower() or "agent stopped" in output_final.lower()

            if es_compleja:
                st.session_state.ultima_pregunta_compleja = pregunta
                st.session_state.sugerencias_mostradas = True
                output_final = (
                    "🔄 **¡Buenas noticias! Tienes una consulta muy rica en datos.** 🧠\n\n"
                    "Pero es tan completa que requiere dividirla para no perderse en el camino. "
                    "He preparado **3 subpreguntas** abajo 👇 que puedes responder una a una."
                )

            st.session_state.mensajes.append({"role": "assistant", "content": output_final})
            with st.chat_message("assistant"):
                st.markdown(output_final)

            if not es_compleja:
                comuna_detectada = detectar_comuna_en_texto(pregunta)

                if comuna_detectada is None and "mapa" in pregunta.lower() and st.session_state.titulo_mapa:
                    partes = st.session_state.titulo_mapa.split(" en ")
                    if len(partes) > 1:
                        comuna_detectada = partes[-1]

                if comuna_detectada and "Auditoría Final" in diccionario_dfs:
                    df_m = diccionario_dfs["Auditoría Final"]
                    if 'comuna' in df_m.columns:
                        df_comuna = df_m[df_m['comuna'].astype(str).str.strip().str.lower() == comuna_detectada.strip().lower()]
                        
                        if 'RBD' in df_comuna.columns:
                            df_comuna = df_comuna.drop_duplicates(subset=['RBD'])

                        if 'LATITUD' in df_comuna.columns and 'LONGITUD' in df_comuna.columns:
                            df_mapa = df_comuna[['LATITUD', 'LONGITUD', 'Nombre_Colegio']].dropna()
                            if not df_mapa.empty:
                                df_mapa = df_mapa.rename(columns={'LATITUD': 'lat', 'LONGITUD': 'lon'})
                                st.session_state.df_mapa = df_mapa
                                st.session_state.titulo_mapa = f"🗺️ Ubicación de Establecimientos en {comuna_detectada.title()}"
                                st.rerun()

        except Exception as e:
            error_msg = str(e)
            st.error(f"⚠️ Error en el agente: {error_msg[:200]}")

            if "max iterations" in error_msg.lower():
                st.session_state.ultima_pregunta_compleja = pregunta
                st.session_state.sugerencias_mostradas = True
                with st.chat_message("assistant"):
                    st.markdown("🔄 **Consulta muy completa.** Revisa las sugerencias abajo 👇 para dividirla.")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"💥 Algo salió mal: {error_msg[:100]}. ¿Puedes reformular?")