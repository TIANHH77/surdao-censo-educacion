import pandas as pd

# Rutas
ruta_matriz = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\matriz_colegios_con_comunas.parquet"
ruta_historico = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\historico_volatilidad_riesgo.parquet"
ruta_salida = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_completo_auditoria.parquet"

# Cargar
df_matriz = pd.read_parquet(ruta_matriz)
df_historico = pd.read_parquet(ruta_historico)

# --- NORMALIZACIÓN ---
# Convertimos RBD a string en ambos para que coincidan perfectamente
df_matriz['RBD'] = df_matriz['RBD'].astype(str)
df_historico['RBD'] = df_historico['RBD'].astype(str)

# --- FUSIÓN ---
# Fusionamos por RBD y Anio. 'left' mantiene toda la matriz original
df_final = pd.merge(
    df_matriz, 
    df_historico[['RBD', 'Anio', 'Volatilidad_Rendimiento']], 
    on=['RBD', 'Anio'], 
    how='left'
)

# --- VALIDACIÓN POST-FUSIÓN ---
print(f"Filas originales: {len(df_matriz)}")
print(f"Filas tras fusión: {len(df_final)}")
print(f"Datos de volatilidad encontrados: {df_final['Volatilidad_Rendimiento'].notnull().sum()}")

# Guardar
df_final.to_parquet(ruta_salida, index=False)
print(f"✅ Dataset completo guardado en: {ruta_salida}")