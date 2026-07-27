"""
Bot de Telegram para Sur DAO — Versión Definitiva (Pura Conversación)
- Sin botones inline (evita errores de timeout en Telegram)
- Mensaje de bienvenida con flujo de ejemplo real (Caso Talagante)
- Token seguro por variable de entorno
- Caché de respuestas con TTL de 24 horas
- Paracaídas anti-error de Markdown
"""
import os
import hashlib
from time import time
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

from agent_core import create_surdao_agent

# 🔒 SEGURIDAD: Exigir token por entorno, sin hardcodear
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ ERROR CRÍTICO: La variable de entorno 'TELEGRAM_BOT_TOKEN' no está configurada.")

# ──────────────────────────────────────────────
# CARGA DEL DATAMART
# ──────────────────────────────────────────────
def cargar_datamart_completo():
    dfs = {}
    ruta_carpeta = "data"

    print(f"🚀 Iniciando carga masiva desde: {ruta_carpeta}")

    if not os.path.exists(ruta_carpeta):
        print(f"❌ ¡Alerta! La carpeta '{ruta_carpeta}' no existe.")
        return dfs

    nombres_especiales = {
        "educacion_historico_2012_2023.parquet": "Histórico Educativo (2012-2023)",
        "educacion_censo_2024.parquet": "Matriz 2024 + Censo",
        "dataset_auditoria_final.parquet": "Auditoría Final"
    }

    archivos = os.listdir(ruta_carpeta)
    archivos_parquet = [f for f in archivos if f.endswith('.parquet')]

    for archivo in archivos_parquet:
        ruta_completa = os.path.join(ruta_carpeta, archivo)

        if archivo in nombres_especiales:
            nombre_tabla = nombres_especiales[archivo]
        else:
            nombre_tabla = archivo.replace('.parquet', '').replace('_', ' ')

        try:
            df = pd.read_parquet(ruta_completa)
            df = df.drop_duplicates()
            dfs[nombre_tabla] = df
        except Exception as e:
            print(f"⚠️ Error crítico cargando {archivo}: {e}")

    print(f"\n🔥 DATAMART LISTO: {len(dfs)} tablas activas en memoria RAM.")
    return dfs

print("Cargando datos locales...")
dfs = cargar_datamart_completo()

print("Datos cargados. Construyendo agente...")
agente = create_surdao_agent(dfs)
print("Agente listo. Iniciando bot de Telegram...")

# ──────────────────────────────────────────────
# ESTADO GLOBAL & CACHÉ CON TTL
# ──────────────────────────────────────────────
historiales: dict[int, list] = {}
cache_respuestas: dict[str, tuple[str, float]] = {}

def hash_pregunta(pregunta: str) -> str:
    return hashlib.md5(pregunta.lower().strip().encode()).hexdigest()

def obtener_cache(pregunta: str) -> str | None:
    h = hash_pregunta(pregunta)
    if h in cache_respuestas:
        resp, ts = cache_respuestas[h]
        if time() - ts < 86400:  # Vigencia de 24 horas
            return resp
        else:
            del cache_respuestas[h]
    return None

def guardar_cache(pregunta: str, respuesta: str):
    h = hash_pregunta(pregunta)
    cache_respuestas[h] = (respuesta, time())

# ──────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────
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
            sub.append(f"📊 ¿Cuál es la **{tema}** de esa comuna?")
    if not sub:
        sub = ["📊 ¿Cuánta población tiene?", "📊 ¿Cuál es el nivel educativo?", "📊 ¿Hay datos de inmigración?"]
    return sub[:3]

async def enviar_con_paracaidas(update_or_query, texto: str):
    """Envía el mensaje intentando Markdown; si Telegram rechaza el formato, lo manda en texto plano."""
    target = update_or_query.message if hasattr(update_or_query, "message") else update_or_query
    try:
        await target.reply_text(texto, parse_mode="Markdown")
    except Exception as e:
        print(f"⚠️ Paracaídas activado. Falló Markdown: {e}")
        await target.reply_text(texto)

# ──────────────────────────────────────────────
# HANDLERS
# ──────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje_bienvenida = (
        "🤖 ¡Hola! Soy **Sur DAO 2.0**, tu Agente Analítico Especializado en Educación y Sociodemografía de Chile. 🇨🇱📊\n\n"
        "Tengo en mi memoria un datamart avanzado con más de **60 bases de datos cruzadas** (Censo 2024 y MINEDUC).\n\n"
        "💡 **¿Cómo sacarme el mayor provecho?**\n"
        "Te recomiendo ir de lo general a lo particular. Aquí tienes un flujo que funciona perfecto:\n\n"
        "1️⃣ **Parte por tu comuna:**\n"
        "👉 *'¿Qué datos tienes de Talagante?'*\n\n"
        "2️⃣ **Haz zoom en un tema:**\n"
        "👉 *'¿Cómo es la educación de los inmigrantes comparada con los nacionales ahí?'*\n\n"
        "3️⃣ **Cruza con evolución histórica:**\n"
        "👉 *'Respecto a la evolución del rendimiento y el ratio estudiantes/profesor, ¿qué datos históricos tienes?'*\n\n"
        "También puedes usar `/buscar [tema]` para encontrar tablas en mi base de datos.\n\n"
        "¿Por qué comuna empezamos a analizar hoy? 🚀"
    )
    await enviar_con_paracaidas(update, mensaje_bienvenida)

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    pregunta = update.message.text
    chat_history = historiales.get(chat_id, [])

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        output_final = obtener_cache(pregunta)
        
        if not output_final:
            respuesta = agente.invoke({"input": pregunta, "chat_history": chat_history})
            output_final = respuesta["output"]

            if "stopped due to max iterations" in output_final.lower() or "agent stopped" in output_final.lower():
                subpreguntas = generar_subpreguntas(pregunta)
                output_final = (
                    "🔄 **¡Esa pregunta abarca mucha información!** Necesito dividirla para responderte con precisión.\n\n"
                    "Puedes reformular tu pregunta probando con opciones como estas 👇\n\n"
                )
                for i, sp in enumerate(subpreguntas, 1):
                    output_final += f"{i}. {sp}\n"
                output_final += "\n💡 *Ejemplo: '¿Cuál es el nivel educativo de Talagante?'*"
            else:
                guardar_cache(pregunta, output_final)

        # Paracaídas de formato
        await enviar_con_paracaidas(update, output_final)

    except Exception as e:
        error_msg = str(e)
        output_final = f"⚠️ Ocurrió un error procesando tu consulta:\n\n{error_msg[:300]}"
        await update.message.reply_text(output_final)

    chat_history.append(HumanMessage(content=pregunta))
    chat_history.append(AIMessage(content=output_final))
    historiales[chat_id] = chat_history[-20:]

async def buscar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("🔍 Usa: /buscar [palabra clave]\nEj: /buscar discapacidad")
        return

    palabra = " ".join(context.args)
    encontradas = []
    for nombre, df in dfs.items():
        if palabra.lower() in nombre.lower() or any(palabra.lower() in str(col).lower() for col in df.columns):
            encontradas.append(f"📁 **{nombre[:50]}** ({df.shape[0]:,} filas)")

    if encontradas:
        msg = f"🔎 Tablas relacionadas con '{palabra}':\n\n" + "\n".join(encontradas[:8])
        if len(encontradas) > 8:
            msg += f"\n\n...y {len(encontradas)-8} más."
    else:
        msg = f"❌ No encontré tablas relacionadas con '{palabra}'."

    await enviar_con_paracaidas(update, msg)

async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 **Comandos de Sur DAO:**\n\n"
        "/start — Mensaje de bienvenida con flujo de ejemplo\n"
        "/buscar [tema] — Busca tablas por tema\n"
        "/ayuda — Muestra esta ayuda\n\n"
        "**Ejemplos de análisis conversacional:**\n"
        "• *'¿Qué datos tienes de Sierra Gorda?'*\n"
        "• *'Dame el detalle sobre la migración interna ahí'*\n"
        "• *'Cruza el promedio de notas con el nivel educativo en Talagante'*\n"
        "• *'¿Qué colegios tienen la mejor nota?'*"
    )
    await enviar_con_paracaidas(update, msg)

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("buscar", buscar))
    app.add_handler(CommandHandler("ayuda", ayuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

    print("🤖 Bot Sur DAO corriendo (Pura Conversación). Ctrl+C para detener.")
    app.run_polling()

if __name__ == "__main__":
    main()