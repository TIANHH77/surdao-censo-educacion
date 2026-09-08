# 🏔️ Sur DAO 2.0: Agente Analítico Cívico

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agentes_ReAct-green)
![Data](https://img.shields.io/badge/Datamart-11_Bases_Maestras-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI_Compatible_%2F_Local-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit_%2B_Telegram-red)

**Sur DAO 2.0** es un agente conversacional autónomo diseñado para democratizar el acceso a datos sociodemográficos y educativos de Chile. A diferencia de un chatbot tradicional, este agente **no responde con texto pre-entrenado**: razona la pregunta, localiza las tablas relevantes mediante búsqueda híbrida, y **escribe y ejecuta código Python (pandas) en memoria** para extraer la respuesta estadística exacta.

> 🏆 **Desafío Alura Latam / Oracle Next Education (ONE)**
> **Nota sobre la escala:** La versión actual de este repositorio utiliza un Datamart consolidado en **11 Bases Maestras** para permitir su ejecución ágil en la nube (Streamlit Cloud). Sin embargo, el *Data Lake* original y los pipelines (ETL) que alimentan este ecosistema procesan y cruzan más de **510 millones de registros estatales** de la trayectoria educativa e histórica de Chile (2012-2025).

🌐 [Ver Aplicación en Vivo (Streamlit Cloud)](https://surdao-censo-educacion.streamlit.app/) • 🤖 [Probar Bot de Telegram (@SurdaoBot)](https://t.me/SurdaoBot)

---

## 🏗️ Estructura del Datamart (`/data`)

El proyecto ingiere archivos Parquet ultralivianos y fuertemente tipados. Para esta versión, los datos masivos fueron destilados en una super-tabla de auditoría y 10 tablas demográficas clave:

```text
surdao-censo-educacion/
├── app.py                  # Centro de Mando Web (Streamlit)
├── telegram_bot.py         # Bot asíncrono
├── agent_core.py           # Cerebro LangChain (RAG + AST REPL)
└── data/
    ├── manual_uso_microdatos_censo2024.pdf   # Base de conocimiento (RAG)
    ├── 📂 CENSO_2024/
    │   ├── D2_2_Población_censada_por_tramo_de_edad...parquet
    │   ├── D4_4_Inmigrantes_internacionales_por_país...parquet
    │   ├── D5_2_Población_censada_por_comuna_hace_5_años...parquet
    │   ├── P1_2_Población...con_discapacidad...parquet
    │   ├── P2_2_Población...pueblo_originario...parquet
    │   ├── P7_2_Población_según_nivel_educativo...parquet
    │   ├── P7_4_Años_de_escolaridad_promedio_comuna...parquet
    │   ├── P7_8_Tasa_de_asistencia_neta...parquet
    │   ├── P7_10_Población...sabe_leer_o_escribir...parquet
    │   └── P8_2_Años_de_escolaridad_población_inmigrante...parquet
    └── 📂 educacion/
        └── dataset_auditoria_final.parquet   # Super-tabla MINEDUC
```

## 🚀 Características Principales de Ingeniería

- 🧠 **Ejecución de código dinámico (AST REPL):** el LLM genera código pandas sobre la marcha. Cada ejecución corre en un `ThreadPoolExecutor` con timeout de 30s para evitar que una consulta pesada cuelgue al agente. Se inyecta en el entorno una utilidad de normalización de texto (`normalizar()`) para emparejar nombres de comuna sin errores por tildes o mayúsculas.
- 🛡️ **Arquitectura resiliente (fallback):** el agente arma una cascada de modelos de nube (Claude 3.5 Haiku, GPT-4o-mini, Mistral Small) vía OpenRouter, y cae a un modelo local (`groq/openai/gpt-oss-120b`) cuando no hay API key configurada.
- 🔍 **Búsqueda híbrida de metadatos:** antes de escribir código, el agente usa un índice semántico (FAISS + `all-MiniLM-L6-v2`) para descubrir qué base maestra del datamart necesita consultar.
- 📚 **RAG sobre el Manual del Censo 2024:** un retriever separado responde dudas metodológicas (definiciones, fórmulas, tratamiento de valores especiales) citando el manual oficial.
- 🗺️ **Mapeo geoespacial automático:** la interfaz web detecta comunas mencionadas en la conversación y cruza la Auditoría Final para renderizar mapas interactivos (`pydeck`) con las coordenadas de los establecimientos.
- ⚡ **Caché de respuestas (Telegram):** las respuestas se cachean por hash de la pregunta con un TTL de 24 horas para ahorrar llamadas al LLM en preguntas repetidas.

> 🚧 **En progreso:** el abort automático ante errores fatales de API (auth/billing/rate-limit) y el anclaje de la caché de Telegram al `chat_id` del usuario están diseñados pero aún no completamente implementados — ver sección **Roadmap** abajo.

## ⚙️ Stack Tecnológico

| Capa | Tecnología | Propósito |
|---|---|---|
| Orquestación | LangChain (`create_tool_calling_agent`) | Razonamiento tipo ReAct y selección de herramientas |
| Procesamiento | Pandas + `PythonAstREPLTool` | Motor de cálculo relacional en RAM |
| RAG & Búsqueda | FAISS + `all-MiniLM-L6-v2` | Embeddings para el Manual del Censo y el índice de tablas |
| Interfaces | Streamlit / `python-telegram-bot` | UI web con geolocalización / bot con polling asíncrono |
| Almacenamiento | Parquet (columnar) | Compresión y lectura ultrarrápida |

## 🚀 Instalación y Despliegue Local

⚠️ **Requisito crítico:** Python 3.11 (versiones superiores, como 3.12+, presentan incompatibilidades de dependencias con LangChain y provocan errores de importación de agentes).

```bash
# 1. Clonar el repositorio
git clone https://github.com/tianhh77/surdao-censo-educacion.git
cd surdao-censo-educacion

# 2. Crear y activar entorno virtual estrictamente en Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate      # En Windows
# source venv/bin/activate # En Linux/Mac

# 3. Instalar dependencias exactas
pip install -r requirements.txt

# 4. Configurar variables de entorno (.env)
# Crea un archivo .env en la raíz con:
TELEGRAM_BOT_TOKEN="tu_token_de_telegram"
OPENROUTER_API_KEY="tu_api_key_de_openrouter"

# 5. Ejecutar los servicios
streamlit run app.py       # Terminal 1: Inicia el Centro de Mando Web
python telegram_bot.py     # Terminal 2: Inicia el Bot de Telegram asíncrono
```

## 🧭 Limitaciones conocidas / Roadmap

- [ ] Hacer que el error fatal de API (auth/billing) corte la cascada de fallback en vez de solo loguearlo.
- [ ] Anclar la caché de Telegram al `chat_id` (y no solo al texto de la pregunta) para no cruzar respuestas entre usuarios distintos.
- [ ] Mover la llamada al agente en `telegram_bot.py` a un hilo aparte (`asyncio.to_thread`) para no bloquear el event loop durante consultas largas.
- [ ] Persistir los índices FAISS a disco (`save_local` / `load_local`) para acelerar el arranque en frío.
- [ ] Unificar la carga del datamart y la generación de subpreguntas entre `app.py` y `telegram_bot.py` en un módulo compartido.

## 📄 Licencia

*( MIT.)*

---

Desarrollado como desafío arquitectónico y de datos cívicos. Construido entre jornadas de logística urbana. 🚲💻
