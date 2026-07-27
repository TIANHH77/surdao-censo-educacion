import os
import pandas as pd

# Lista específica de los archivos que forman parte del núcleo analítico
archivos_clave = [
    "educacion_historico_2012_2023.parquet",
    "educacion_censo_2024.parquet",
    "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet",
    "P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet",
    "P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet"
]

data_dir = "data"
catalogo_md = "# Catálogo de Datos Seleccionados (Educación y Censo)\n\n"

print("🔍 Analizando estructuras de los datasets clave...")

for archivo in archivos_clave:
    ruta = os.path.join(data_dir, archivo)
    if os.path.exists(ruta):
        try:
            df = pd.read_parquet(ruta)
            columnas = list(df.columns)
            
            # Detección segura con bucle explícito
            tiene_rbd = False
            for col in columnas:
                if 'rbd' in str(col).lower():
                    tiene_rbd = True
                    break
            
            tiene_comuna = False
            for col in columnas:
                if any(c in str(col).lower() for c in ['comuna', 'codigo_comuna']):
                    tiene_comuna = True
                    break
            
            catalogo_md += f"## Archivo: {archivo}\n"
            catalogo_md += f"- **Total Filas:** {df.shape[0]:,}\n"
            catalogo_md += f"- **RBD detectado:** {tiene_rbd}\n"
            catalogo_md += f"- **Comuna/CUT detectado:** {tiene_comuna}\n"
            catalogo_md += "- **Columnas:**\n"
            for col in columnas:
                catalogo_md += f"  - {col}\n"
            catalogo_md += "\n"
            print(f"[OK] Procesado: {archivo}")
        except Exception as e:
            print(f"[ERROR] No se pudo leer {archivo}: {e}")
    else:
        print(f"[ADVERTENCIA] El archivo no existe en la ruta: {ruta}")

# Guardar el catálogo generado
output_path = "catalogo_educacion_seleccion.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(catalogo_md)

print(f"\n✨ ¡Listo! Catálogo reducido generado con éxito en: {output_path}")