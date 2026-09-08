🤖 Agente Auditor Educativo - Sur DAODescripción del ProyectoEste agente de IA, desarrollado para el challenge Alura Agente, actúa como un consultor experto capaz de auditar grandes volúmenes de datos educativos y censales. 

El agente permite realizar consultas en lenguaje natural sobre rendimiento escolar, asistencia y demografía, evitando la necesidad de navegar manualmente entre cientos de archivos.  

Arquitectura TécnicaLenguaje: Python.  Orquestación: LangChain con arquitectura de Tools modulares.  

Procesamiento: Pandas sobre archivos .parquet (optimizando el rendimiento para grandes volúmenes de datos).  

Motor: Llama 3.3 70B vía Groq.

Resolución de Entidades: Sistema jerárquico de búsqueda (CUT/RBD) para garantizar precisión en los datos.

Ejemplos de ConsultasEl agente está entrenado para responder preguntas complejas como:

"¿Cuál es la evolución del promedio de notas en la comuna de Isla de Maipo (CUT 13401) entre 2012 y 2024?"

"¿Existe una correlación entre el nivel educativo 'Nunca Asistió' (Censo 2024) y el rendimiento en los colegios de la comuna?


"Instrucciones de EjecuciónClona este repositorio.Configura tu archivo .env con las credenciales necesarias (Groq API Key).Instala las dependencias: pip install -r requirements.txt.Ejecuta la interfaz: streamlit run agente_app.py.Deploy (Implementación)Estado: Aplicación en línea / Despliegue funcional.  Enlace de acceso: [Inserta aquí tu link de OCI o Streamlit Cloud]