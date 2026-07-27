import pandas as pd

print("🚀 Iniciando la división temporal y el cruce para Sur DAO...")

# ==========================================
# 1. CARGAR LA MATRIZ PRINCIPAL
# ==========================================
print("Cargando matriz de educación...")
df_educacion = pd.read_parquet("data/dataset_auditoria_final.parquet")

# ==========================================
# 2. SEPARAR POR AÑO
# ==========================================
df_hist = df_educacion[df_educacion['Anio'] < 2024].copy()
df_2024 = df_educacion[df_educacion['Anio'] == 2024].copy()

print(f"📊 Datos Históricos (2012-2023): {df_hist.shape[0]} filas")
print(f"📊 Datos 2024: {df_2024.shape[0]} filas")

df_hist.to_parquet("data/educacion_historico_2012_2023.parquet", index=False)
print("💾 Archivo histórico guardado.")

# ==========================================
# 3. NORMALIZACIÓN INTELIGENTE DE LLAVES
# ==========================================
def normalizar_comuna(df):
    # 1. Eliminar columnas duplicadas de origen si las hay
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Encontrar la columna que tiene el código
    col_codigo = None
    if "codigo_comuna" in df.columns:
        col_codigo = "codigo_comuna"
    elif "CUT" in df.columns:
        col_codigo = "CUT"
    elif "comuna" in df.columns:
        col_codigo = "comuna"
    
    # 3. Renombrar a la llave maestra si es necesario
    if col_codigo and col_codigo != "codigo_comuna":
        df = df.rename(columns={col_codigo: "codigo_comuna"})
    
    # 4. Asegurarnos que no existan gemelos post-renombramiento y forzar Int64
    if "codigo_comuna" in df.columns:
        df = df.loc[:, ~df.columns.duplicated()]
        df["codigo_comuna"] = pd.to_numeric(df["codigo_comuna"], errors="coerce").astype("Int64")
    
    return df

print("Cargando y normalizando bases del Censo 2024...")
df_alfabetizacion = normalizar_comuna(pd.read_parquet("data/P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet"))
df_nivel_educ = normalizar_comuna(pd.read_parquet("data/P7_2_Población_según_nivel_educativo_más_alto_alcanzado_según_c.parquet"))
df_asistencia = normalizar_comuna(pd.read_parquet("data/P7_8_Tasa_de_asistencia_neta_por_nivel_educativo_según_comuna.parquet"))

# Asegurar que la matriz 2024 también pase por el mismo escáner
df_2024 = normalizar_comuna(df_2024)

# ==========================================
# 4. EL CRUCE MAESTRO (SOLO 2024)
# ==========================================
print("Cruzando datos educativos 2024 con Censo...")
df_2024_censo = pd.merge(df_2024, df_alfabetizacion, on="codigo_comuna", how="left", suffixes=('', '_alfabet'))
df_2024_censo = pd.merge(df_2024_censo, df_nivel_educ, on="codigo_comuna", how="left", suffixes=('', '_nivel'))
df_2024_censo = pd.merge(df_2024_censo, df_asistencia, on="codigo_comuna", how="left", suffixes=('', '_asistencia'))

# Eliminar duplicados finales
df_2024_censo = df_2024_censo.loc[:, ~df_2024_censo.columns.duplicated()]

# ==========================================
# 5. EXPORTACIÓN FINAL
# ==========================================
ruta_2024 = "data/educacion_censo_2024.parquet"
df_2024_censo.to_parquet(ruta_2024, index=False)
print(f"✅ ¡Éxito! Tabla 2024 enriquecida guardada en: {ruta_2024}")