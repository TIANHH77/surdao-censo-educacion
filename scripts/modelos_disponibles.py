import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carga tus variables de entorno
load_dotenv()

# Configura la API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Lista los modelos que soportan generación de contenido
print("Modelos disponibles para tu cuenta:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")