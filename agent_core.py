import os
import sys
import re
import json
import unicodedata
import pandas as pd
import concurrent.futures
from typing import List

# === CARGA DE VARIABLES DE ENTORNO ===
from dotenv import load_dotenv
load_dotenv()
# =====================================

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
    if not isinstance(texto, str):
        return str(texto)
    
    texto = texto.lower().strip()
    reemplazos = [
        ('á','a'),('à','a'),('ä','a'),('â','a'),
        ('é','e'),('è','e'),('ë','e'),('ê','e'),
        ('í','i'),('ì','i'),('ï','i'),('î','i'),
        ('ó','o'),('ò','o'),('ö','o'),('ô','o'),
        ('ú','u'),('ù','u'),('ü','u'),('û','u')
    ]
    for orig, rem in reemplazos:
        texto = texto.replace(orig, rem)
        
    texto = re.sub(r'[^a-z0-9\sñ]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto

# ============================================================
#  1. CONFIGURACIÓN CENTRALIZADA DE MODELOS
# ============================================================

MODELOS_NUBE_FALLBACK = [
    "openrouter/free",
    "google/gemma-3-12b:free",
    "qwen/qwen-2.5-7b-instruct:free"
]

MODELO_LOCAL = "groq/openai/gpt-oss-120b"

def _get_env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            pass
    return val

def get_llms() -> List[ChatOpenAI]:
    openrouter_key = _get_env("OPENROUTER_API_KEY")
    openrouter_base = _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if openrouter_key:
        print("🌐 Modo activo: NUBE (OpenRouter) con fallback por reintento (Modelos Gratuitos)")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
ERRORES_CUENTA = [
    "authentication", "unauthorized", "invalid api key",
    "incorrect api key", "insufficient_quota", "billing", "payment",
]

ERRORES_MODELO = [
    "rate limit exceeded", "invalid model", "model not found", "not a valid model",
]

def _clasificar_error(error: Exception) -> str:
    msg = str(error).lower()
    if any(e in msg for e in ERRORES_CUENTA):
        return "cuenta"
    if any(e in msg for e in ERRORES_MODELO):
        return "modelo"
    return "desconocido"

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
            tipo_error = _clasificar_error(e)
            
            if tipo_error == "cuenta":
                print(f"🚫 Modelo #{i+1}: Error crítico de CUENTA ({type(e).__name__}). Se aborta la cadena para no gastar de más.")
                return {"output": "⚠️ Error crítico de conexión o cuota con el proveedor de IA. Revisa tus credenciales."}
            
            print(f"⚠️ Modelo #{i+1} falló por error de {tipo_error.upper()} ({type(e).__name__}). Probando el siguiente modelo...")
            continue

    return {
        "output": (
            f"⚠️ Todos los modelos disponibles fallaron en cadena.\n"
            f"Último error: {type(ultimo_error).__name__}"
        )
    }

# ============================================================
# 4.5 CONFIGURACIÓN DEL PERFIL CENSAL MULTIDIMENSIONAL
# ============================================================
PERFIL_CENSAL_CONFIG = {
    "escolaridad_promedio": {
        "prefijo": "P7_4",
        "columnas": ["sexo", "años_de_escolaridad_promedio"],
        "filtro_columna": "sexo",
        "filtro_valor": "Total Comuna", 
    },
    "envejecimiento": {
        "prefijo": "D2_2",
        "columnas": ["sexo", "0_14", "15_64", "65_años_o_más", "indice_de_envejecimiento"],
        "filtro_columna": "sexo",
        "filtro_valor": "Total", 
    },
    "inmigracion_paises": {
        "prefijo": "D4_4",
        "columnas": ["pais_o_continente_de_nacimiento", "inmigrantes_internacionales"],
        "filtro_columna": None, 
        "filtro_valor": None,
    },
    "discapacidad": {
        "prefijo": "P1_2",
        "columnas": ["grupos_de_edad", "poblacion_de_5_años_o_más_con_discapacidad"],
        "filtro_columna": "grupos_de_edad",
        "filtro_valor": "Total", 
    },
    "pueblos_originarios": {
        "prefijo": "P2_2",
        "columnas": ["mapuche", "aymara", "rapa_nui", "pueblo_no_declarado"],
        "filtro_columna": None,
        "filtro_valor": None,
    },
    "maternidad_y_familia": {
        "prefijo": "D6_2",
        "columnas": ["hijos_e_hijas_declarados", "cantidad_de_hijos_e_hijas", "0", "1", "2", "3", "4"],
        "filtro_columna": "hijos_e_hijas_declarados",
        "filtro_valor": "Total",
    }
}

def obtener_perfil_censal(cut_comuna: float, dfs: dict) -> dict:
    """Extrae múltiples dimensiones sociales de una comuna con búsqueda dinámica de llaves."""
    perfil = {}
    for etiqueta, cfg in PERFIL_CENSAL_CONFIG.items():
        prefijo_crudo = cfg["prefijo"]
        # Cubrimos ambas posibilidades por si el loader ya reemplazó los guiones
        variante_espacio = prefijo_crudo.replace('_', ' ')
        
        # Búsqueda dinámica de la llave exacta en memoria
        llave_real = next((k for k in dfs.keys() if k.startswith(prefijo_crudo) or k.startswith(variante_espacio)), None)
        
        if not llave_real:
            perfil[etiqueta] = {"error": f"No se encontró tabla con prefijo '{prefijo_crudo}' en memoria."}
            continue
            
        df = dfs[llave_real]
        
        try:
            # Filtrar por comuna (usando el CUT)
            sub = df[df["codigo_comuna"].astype(float) == cut_comuna]
            
            # Aplicar filtro específico
            if cfg["filtro_columna"] and cfg["filtro_valor"] and cfg["filtro_columna"] in sub.columns:
                sub = sub[sub[cfg["filtro_columna"]] == cfg["filtro_valor"]]
                
            # Extraer solo las columnas solicitadas que existan en el dataframe
            cols_presentes = [c for c in cfg["columnas"] if c in sub.columns]
            perfil[etiqueta] = sub[cols_presentes].to_dict("records")
        except Exception as e:
            # Ahora la consola canta exactamente dónde y por qué falló
            print(f"⚠️ Cruce censal falló en {etiqueta} (tabla: {llave_real}): {e}")
            perfil[etiqueta] = {"error": f"Fallo al procesar: {str(e)}"}
            
    return perfil
# ============================================================
# 5. FÁBRICA DEL AGENTE PRINCIPAL
# ============================================================
def create_surdao_agent(dfs: dict):
    _python_tool = PythonAstREPLTool(locals={
        "pd": pd, 
        "dfs": dfs, 
        "normalizar": normalizar_texto_chile,
        "json": json
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

    # --- NUEVAS TOOLS TERRITORIALES Y TEMPORALES ---
    @tool
    def escanear_comuna_educativa(nombre_comuna: str) -> str:
        """
        ÚTIL PARA INICIAR UNA AUDITORÍA TERRITORIAL O CUANDO EL USUARIO PREGUNTA POR UNA COMUNA.
        Busca todos los colegios de una comuna y cruza el panorama general con el Censo 2024.
        """
        comuna_norm = normalizar_texto_chile(nombre_comuna)
        df_auditoria = dfs.get("Auditoría Final")
        
        if df_auditoria is None:
            return json.dumps({"error": "La tabla 'Auditoría Final' no está cargada."})
            
        mask = df_auditoria['COMUNA'].apply(normalizar_texto_chile) == comuna_norm
        colegios_comuna = df_auditoria[mask]
        
        if colegios_comuna.empty:
            return json.dumps({"error": f"No se encontraron colegios para la comuna '{nombre_comuna}'."})
        
        colegios_recientes = colegios_comuna.sort_values('Anio').groupby('RBD').last().reset_index()
        top_criticos = colegios_recientes.sort_values(by='Ratio_Alumnos_Docente', ascending=False).head(3)
        lista_criticos = top_criticos[['Nombre_Colegio', 'RBD', 'Ratio_Alumnos_Docente']].to_dict('records')
        
        cut_comuna = float(colegios_recientes['CUT'].iloc[0])
        
        # 🔴 IMPLEMENTACIÓN DEL PERFIL CENSAL MULTIDIMENSIONAL
        datos_censo = obtener_perfil_censal(cut_comuna, dfs)

        resultado = {
            "territorio_auditado": comuna_norm,
            "total_colegios_activos": len(colegios_recientes),
            "alertas_sobrecarga_docente_top3": lista_criticos,
            "contexto_social_censo_2024": datos_censo,
            "instruccion_para_ia": "Informa al usuario este panorama general y pregúntale si quiere ver el detalle de alguno de los colegios críticos mencionando su RBD."
        }
        return json.dumps(resultado)

    @tool
    def analizar_colegio_y_entorno(rbd: int) -> str:
        """
        ÚTIL SIEMPRE QUE EL USUARIO PREGUNTE POR EL RENDIMIENTO O CONTEXTO DE UN COLEGIO ESPECÍFICO.
        Requiere el RBD numérico del colegio. Cruza rendimiento histórico con demografía del Censo 2024.
        """
        df_auditoria = dfs.get("Auditoría Final")
        if df_auditoria is None: return json.dumps({"error": "La tabla 'Auditoría Final' no está cargada."})
        
        colegio = df_auditoria[df_auditoria['RBD'].astype(float) == float(rbd)]
        if colegio.empty:
            return json.dumps({"error": f"No se encontró ningún colegio con el RBD {rbd}"})
        
        colegio_reciente = colegio.sort_values(by='Anio', ascending=False).iloc[0]
        cut_comuna = float(colegio_reciente['CUT'])
        
        # 🔴 IMPLEMENTACIÓN DEL PERFIL CENSAL MULTIDIMENSIONAL
        datos_censo = obtener_perfil_censal(cut_comuna, dfs)

        resultado = {
            "colegio": {
                "nombre": colegio_reciente['Nombre_Colegio'],
                "rbd": int(rbd),
                "comuna": colegio_reciente['COMUNA'],
                "coordenadas": [float(colegio_reciente['LATITUD']), float(colegio_reciente['LONGITUD'])]
            },
            "metricas_educativas": {
                "anio_registro": int(colegio_reciente['Anio']),
                "total_alumnos": float(colegio_reciente['Total_Alumnos']),
                "ratio_alumnos_por_docente": float(colegio_reciente['Ratio_Alumnos_Docente']),
                "promedio_notas": float(colegio_reciente['Promedio_Notas']),
                "volatilidad_historica": float(colegio_reciente['Volatilidad_Rendimiento'])
            },
            "contexto_barrial_censo_2024": datos_censo
        }
        return json.dumps(resultado)

    @tool
    def analizar_trayectoria_historica(rbd: int) -> str:
        """
        ÚTIL CUANDO EL USUARIO PREGUNTA POR LA EVOLUCIÓN, HISTORIA O TENDENCIA DE UN COLEGIO EN EL TIEMPO.
        Devuelve el rendimiento y métricas del colegio año por año desde 2012 hasta 2024.
        """
        df_auditoria = dfs.get("Auditoría Final")
        if df_auditoria is None: return json.dumps({"error": "La tabla 'Auditoría Final' no está cargada."})
        
        colegio_hist = df_auditoria[df_auditoria['RBD'].astype(float) == float(rbd)].sort_values('Anio')
        if colegio_hist.empty:
            return json.dumps({"error": f"No hay historial para el RBD {rbd}"})
            
        trayectoria = colegio_hist[['Anio', 'Total_Alumnos', 'Ratio_Alumnos_Docente', 'Promedio_Notas']].to_dict('records')
        
        primer_registro = trayectoria[0]
        ultimo_registro = trayectoria[-1]
        variacion_notas = round(ultimo_registro['Promedio_Notas'] - primer_registro['Promedio_Notas'], 2)
        
        resultado = {
            "colegio": colegio_hist['Nombre_Colegio'].iloc[0],
            "periodo_registrado": f"{primer_registro['Anio']} al {ultimo_registro['Anio']}",
            "variacion_total_notas": variacion_notas,
            "linea_de_tiempo_anual": trayectoria,
            "instruccion_para_ia": "Analiza la tendencia en 'linea_de_tiempo_anual'. Detecta en qué año hubo caídas abruptas de notas y verifica si coinciden con un aumento repentino en el 'Ratio_Alumnos_Docente'."
        }
        return json.dumps(resultado)

    indice_semantico_tablas = construir_indice_tablas(dfs)
    buscar_tablas_en_datamart = crear_tool_busqueda_hibrida(dfs, indice_semantico_tablas)

    herramientas = [
        ejecutar_pandas, 
        buscar_tablas_en_datamart, 
        escanear_comuna_educativa, 
        analizar_colegio_y_entorno, 
        analizar_trayectoria_historica
    ]
    
    rag_tool = get_rag_tool()
    if rag_tool:
        herramientas.append(rag_tool)

    prompt_sistema = """Eres el **Agente Principal de Sur DAO**, un asistente experto en datos sociodemográficos, educativos y censales de Chile, basado rigurosamente en el Manual del Censo 2024.

## 🔧 HERRAMIENTAS DISPONIBLES
1. **`buscar_tablas_en_datamart(palabra_clave)`** → Úsala primero para encontrar las tablas relevantes según el tema o comuna.
2. **`ejecutar_pandas(codigo)`** → Obligatoria para extraer las cifras reales de tablas complejas. Empieza con `df = dfs["Nombre EXACTO"]`.
3. **`consultar_manual_censo(query)`** → Solo para definiciones metodológicas o fórmulas.
4. **`escanear_comuna_educativa(nombre_comuna)`** → Úsala para iniciar auditorías territoriales y obtener un panorama general de los colegios de una comuna cruzados con el Censo.
5. **`analizar_colegio_y_entorno(rbd)`** → Úsala para analizar un colegio específico (por su RBD) cruzado con su entorno social y comunal.
6. **`analizar_trayectoria_historica(rbd)`** → Úsala para ver la evolución y tendencia anual de un colegio (notas vs carga docente) a lo largo del tiempo.

## ⚠️ REGLAS ESTRICTAS Y METODOLÓGICAS (MANUAL CENSO 2024)
1. **PROHIBIDO SER UN AGENTE VAGO:** Si usas una herramienta de búsqueda, TIENES PROHIBIDO limitarte a mostrar nombres. Debes invocar inmediatamente la herramienta correcta para extraer los números y presentar las cifras reales.
2. **CERO INVENTOS:** Si un dato no está en el datamart o la herramienta devuelve un error, di "No disponible". Está prohibido usar datos de ejemplo.
3. **Manejo de Valores Especiales:** Antes de promediar o sumar, DEBES excluir los valores especiales: `-99` (No responde), `-66` (Suprimido por anonimización) y `NA` (No aplica).
4. **Cálculo de Proporciones:** Excluye siempre los casos de "No respuesta" (`-99`) del denominador.
5. **FILTRADO OBLIGATORIO DE TEXTOS:** ES OBLIGATORIO usar la función auxiliar `normalizar()` en AMBOS lados de la igualdad al filtrar columnas de texto como comunas. Respeta y conserva siempre la letra `ñ`.
6. **Filtro de Sexo y Totales:** Las tablas demográficas separan las filas por `sexo`. NUNCA sumes sin filtrar antes explícitamente `df[df['sexo'] == 'Total']` (o equivalente) para evitar duplicar población.
7. **Redondeo:** Todos los indicadores y promedios finales deben presentarse redondeados a un (1) decimal.
8. **PROHIBIDO MODIFICAR DATOS:** Tienes ESTRICTAMENTE PROHIBIDO usar `inplace=True`, borrar columnas originales, o modificar el diccionario `dfs`.
9. **REGLA CRÍTICA PARA EL USO DE PANDAS:** Cuando uses la herramienta `ejecutar_pandas` para mostrar datos de un DataFrame, NUNCA uses `print(df)` ni `df.to_string()`. SIEMPRE debes usar `print(df.to_dict(orient='records'))`. Es obligatorio para evitar el desfase y alucinación de columnas.
10. MICRODATOS VS TABLAS AGREGADAS: Si consultas el manual del Censo y muestra fórmulas en código R con valores numéricos (ej. Sexo 1=Hombre, 2=Mujer), recuerda que tus tablas del datamart ya están procesadas con texto. Adapta cualquier fórmula del manual usando los valores de texto reales de las columnas ('Hombre', 'Mujer', 'Total').

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