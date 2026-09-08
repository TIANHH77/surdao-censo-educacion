FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# --- ESTA LÍNEA ES LA CLAVE ---
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "agente_app.py", "--server.port=8501", "--server.address=0.0.0.0"]