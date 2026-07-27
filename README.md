# 🤖 Sur DAO 2.0 — Motor de Datos Híbrido

Proyecto final para el desafío de **Alura Latam**. Un agente analítico construido con **Streamlit** y **LangChain** que combina dos capacidades sobre datos educativos:

1. **Análisis cuantitativo** sobre series históricas de datos educativos (2012–2023) y la matriz educativa 2024 + Censo, usando pandas.
2. **Consulta metodológica (RAG)** sobre el Manual de Uso de Microdatos del Censo 2024, para responder preguntas sobre definiciones y variables oficiales.

El agente decide de forma autónoma qué herramienta usar según la pregunta: cálculos numéricos → pandas; dudas conceptuales/metodológicas → búsqueda semántica sobre el manual.

## 🧠 Arquitectura

- **Frontend / UI:** Streamlit (`st.chat_input`, `st.chat_message`)
- **Orquestación del agente:** LangChain (`langchain.agents`)
- **Modelo de lenguaje:** configurable vía OpenAI-compatible endpoint (Omniroute) — soporta también Google Gemini, Groq y Ollama según se configure
- **Cálculo sobre datos:** `PythonAstREPLTool` ejecutando pandas contra DataFrames cargados en memoria (Parquet)
- **RAG:** `PyPDFLoader` + `RecursiveCharacterTextSplitter` + `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) + `FAISS`

## 📁 Estructura del proyecto

```
Surdao_MVP/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── educacion_historico_2012_2023.parquet
│   ├── educacion_censo_2024.parquet
│   └── manual_uso_microdatos_censo2024.pdf
└── README.md
```

> ⚠️ La carpeta `data/` **no se versiona** en este repositorio (ver `.gitignore`) por el peso de los archivos. Cada usuario debe colocar sus propios archivos ahí antes de correr la app — ver sección de Fuente de Datos.

## 🚀 Instalación

```bash
# 1. Clonar el repo
git clone https://github.com/<tu-usuario>/Surdao_MVP.git
cd Surdao_MVP

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar los datos en la carpeta data/ (ver Fuente de Datos)

# 5. Correr la app
streamlit run app.py
```

## 📊 Fuente de datos

Datos abiertos oficiales de organismos públicos de Chile:

**Educación (Mineduc — Datos Abiertos):**
- [Matrícula por estudiante](https://datosabiertos.mineduc.cl/matricula-por-estudiante-2/) (2004–2025)
- [Rendimiento académico por estudiante](https://datosabiertos.mineduc.cl/rendimiento-por-estudiante-2/) (2002–2025)
- [Asistencia declarada mensual](https://datosabiertos.mineduc.cl/asistencia-declarada-mensual-2/) (2011–2025)
- [Notas y egresados de enseñanza media](https://datosabiertos.mineduc.cl/notas-y-egresados-de-ensenanza-media/) (2002–2024)
- [Directorio de Establecimientos Educacionales](https://datosabiertos.mineduc.cl/directorio-de-establecimientos-educacionales/) (1992–2025)
- Ver listado completo de fuentes utilizadas en `REFERENCIAS_SURDAO.xlsx` (uso interno, no versionado en el repo)

**Censo 2024 (INE Chile):**
- Manual de uso de microdatos: [completar con el link exacto que usaste de censo2024.ine.gob.cl]

> En toda publicación basada en datos de organismos oficiales de estadística, corresponde citar la fuente primaria de los datos (Mineduc / INE Chile).

## 💬 Ejemplos de uso

- *"¿Cuántos alumnos había matriculados en nivel primario en 2020?"* → el agente usa `ejecutar_pandas`.
- *"Según el manual, ¿cómo se define el déficit habitacional?"* → el agente usa `consultar_manual_censo`.

## 🛠️ Tecnologías

`Python` · `Streamlit` · `LangChain` · `Pandas` · `FAISS` · `Sentence-Transformers` · `PyPDF`

## 👤 Autor

Proyecto desarrollado como desafío final del programa de **Alura Latam**.

---
*Este proyecto trabaja con datos educativos y censales oficiales con fines de análisis y aprendizaje.*
