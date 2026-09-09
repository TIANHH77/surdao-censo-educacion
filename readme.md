# 🏔️ Sur DAO 2.0: Agente Analítico Cívico

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agentes_ReAct-green)
![Data](https://img.shields.io/badge/Datamart-50%2B_Archivos_Parquet-orange)
![LLM](https://img.shields.io/badge/LLM-OpenRouter_%2B_Groq_Multicloud-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit_%2B_Telegram-red)

**Sur DAO 2.0** es un agente conversacional autónomo diseñado para democratizar el acceso a datos sociodemográficos y educativos de Chile. A diferencia de un chatbot tradicional, este agente **no responde con texto pre-entrenado**: razona la pregunta, localiza las tablas relevantes mediante búsqueda híbrida, y **escribe y ejecuta código Python (pandas) en memoria** para extraer la respuesta estadística exacta.

> 🏆 **Desafío Alura Latam / Oracle Next Education (ONE)**
> **Nota sobre la escala:** La versión actual de este repositorio utiliza un Datamart consolidado de **más de 50 archivos Parquet** para permitir su ejecución ágil en la nube (Streamlit Cloud). Sin embargo, el *Data Lake* original y los pipelines (ETL) que alimentan este ecosistema procesan y cruzan más de **510 millones de registros estatales** de la trayectoria educativa e histórica de Chile (2012-2025).

🌐 [Ver Aplicación en Vivo (Streamlit Cloud)](https://surdao-censo-educacion.streamlit.app/) • 🤖 [Probar Bot de Telegram (@SurdaoBot)](https://t.me/SurdaoBot)

---

## 🏗️ Estructura del Datamart (`/data`)

El proyecto ingiere archivos Parquet ultralivianos y fuertemente tipados, estructurados para consultas analíticas de alta velocidad. El núcleo combina el catálogo censal y la planimetría educativa:

```text
surdao-censo-educacion/
├── app.py                  # Centro de Mando Web (Streamlit)
├── telegram_bot.py         # Bot asíncrono
├── agent_core.py           # Cerebro LangChain (RAG, REPL, Fallback, Perfil Censal)
└── data/
    ├── manual_uso_microdatos_censo2024.pdf   # Base de conocimiento metodológica
    ├── columnas_totales.md                   # Catálogo RAG de dimensiones
    ├── 📂 CENSO_2024/
    │   ├── D1_D6... (Demografía, envejecimiento, fertilidad, migración)
    │   ├── P1_P8... (Educación, pueblos originarios, discapacidad)
    │   └── [Más de 40 tablas multidimensionales indexadas semánticamente]
    └── 📂 educacion/
        └── dataset_auditoria_final.parquet   # Super-tabla MINEDUC (Histórico 2012-2024)
```

🚀 Características Principales de Ingeniería

🧠 Motor Híbrido y Perfil Censal Multidimensional: Integra herramientas territoriales que cruzan automáticamente el rendimiento histórico escolar (RBD) con hasta 6 dimensiones del Censo 2024 (escolaridad adulta, envejecimiento, migración y pueblos originarios) en una sola pasada usando lookups dinámicos en memoria.

🛡️ Balanceador de Carga Multicloud (Fallback): Arquitectura resiliente que enruta peticiones entre OpenRouter (modelos gratuitos) y Groq (LPU/Llama 3). Incluye un clasificador inteligente de excepciones que aborta la cadena ante errores de cuenta (auth/billing) pero salta al siguiente nodo disponible ante bloqueos de cuota (rate-limits).

🔍 Búsqueda Semántica Dinámica: Un índice FAISS (all-MiniLM-L6-v2) descubre qué tablas usar. El código implementa resolución dinámica de llaves para esquivar inconsistencias en nombres de archivos truncados o espacios invisibles.

⚙️ Ejecución de Código Robusta (AST REPL): El LLM genera código pandas sobre la marcha. Incluye blindaje de tipos de datos (.astype(float)) para cruces exactos de IDs espaciales, y un ThreadPoolExecutor con timeout de 30s para evitar cuelgues.

📚 RAG Metodológico Estricto: Recuperador contextual (k=5) sobre el Manual del Censo 2024 que impone reglas de oro: exclusión de valores -99/-66, filtros de texto normalizados y adaptaciones de microdatos a tablas agregadas.

⚡ Caché de respuestas (Telegram): Las respuestas se cachean por hash de la pregunta con un TTL de 24 horas para ahorrar tokens en consultas repetidas.


| Capa | Tecnología | Propósito |
|---|---|---|
| Orquestación | LangChain (`create_tool_calling_agent`) | Razonamiento ReAct y orquestación de tools |
| Procesamiento | Pandas + `PythonAstREPLTool` | Motor de cálculo relacional en RAM |
| RAG & Búsqueda | FAISS + HuggingFace (`all-MiniLM-L6-v2`) | Embeddings de catálogos y manuales |
| Modelos de Lenguaje | OpenRouter APIs + Groq Cloud | Fallback multicloud para alta disponibilidad |
| Interfaces | Streamlit / `python-telegram-bot` | UI web analítica / Bot con polling asíncrono |


🚀 Instalación y Despliegue Local
⚠️ Requisito crítico: Python 3.11 (versiones superiores, como 3.12+, presentan incompatibilidades de dependencias con LangChain y provocan errores de importación de agentes).

```
# 1. Clonar el repositorio
git clone [https://github.com/tianhh77/surdao-censo-educacion.git](https://github.com/tianhh77/surdao-censo-educacion.git)
cd surdao-censo-educacion

# 2. Crear y activar entorno virtual estrictamente en Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate      # En Windows
# source venv/bin/activate # En Linux/Mac

# 3. Instalar dependencias exactas
pip install -r requirements.txt

# 4. Configurar variables de entorno (.env)
# Crea un archivo .env en la raíz con tus llaves de respaldo:
TELEGRAM_BOT_TOKEN="tu_token_de_telegram"
OPENROUTER_API_KEY="tu_api_key_de_openrouter"
GROQ_API_KEY="tu_api_key_de_groq"

# 5. Ejecutar los servicios
streamlit run app.py       # Terminal 1: Inicia el Dashboard Web
python telegram_bot.py     # Terminal 2: Inicia el Bot de Telegram asíncrono
```


🧭 Limitaciones conocidas / Roadmap
[ ] Anclar la caché de Telegram al chat_id (y no solo al texto de la pregunta) para no cruzar respuestas entre usuarios distintos.
[ ] Mover la llamada al agente en telegram_bot.py a un hilo aparte (asyncio.to_thread) para no bloquear el event loop durante consultas largas.
[ ] Persistir los índices FAISS a disco (save_local / load_local) para acelerar el arranque en frío en Streamlit Cloud.
[ ] Implementar la interfaz de pantalla dividida (Dashboard + Copilot) en Streamlit usando columnas responsivas.

📄 Licencia
MIT License
Copyright (c) 2026 SURDAO
