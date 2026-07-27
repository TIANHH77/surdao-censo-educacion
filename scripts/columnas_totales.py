import os
import pandas as pd

# Rutas de tus archivos (ajusta la carpeta si es necesario)
archivos_mapa = {
    "Histórico Educativo (2012-2023)": "educacion_historico_2012_2023.parquet",
    "Matriz 2024 + Censo": "educacion_censo_2024.parquet",
    "Alfabetización Comunal (P7_10)": "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet",
    "Nivel Educativo Comunal (P7_2)": "P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet",
    "Asistencia Neta Comunal (P7_8)": "P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet",
    "Años Escolaridad (P7_4)": "P7_4_Años_de_escolaridad_promedio_según_sexo_y_comuna.parquet",
    "Envejecimiento (D2_2)": "D2_2_Población_censada_por_tramo_de_edad_e_índice_de_envejecimi.parquet",
    "Discapacidad (P1_2)": "P1_2_Población_de_5_años_o_más_con_discapacidad_por_sexo_según_.parquet",
    "Pueblos Originarios (P2_2)": "P2_2_Población_que_es_o_se_considera_perteneciente_a_un_pueblo_.parquet",
    "Escolaridad Inmigrantes (P8_2)": "P8_2_Años_de_escolaridad_promedio_para_la_población_inmigrante_.parquet",
    "Inmigrantes por País (D4_4)": "D4_4_Inmigrantes_internacionales_por_país_de_nacimiento_según_c.parquet",
    "Migración Interna (D5_2)": "D5_2_Población_censada_por_comuna_de_residencia_habitual_hace_5.parquet"
}

print("🔍 INICIO DE INSPECCIÓN DE DATAMART\n" + "="*50)

for nombre, archivo in archivos_mapa.items():
    ruta = os.path.join("data", archivo) # O pon la ruta exacta de tu carpeta
    if os.path.exists(ruta):
        df = pd.read_parquet(ruta)
        print(f"\n📂 TABLA: {nombre}")
        print(f"   - Archivo: {archivo}")
        print(f"   - Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        print(f"   - Columnas exactas:")
        for col in df.columns:
            print(f"     • {col}")
    else:
        print(f"\n❌ No se encontró: {archivo}")

print("\n" + "="*50 + "\n🏁 INSPECCIÓN FINALIZADA")