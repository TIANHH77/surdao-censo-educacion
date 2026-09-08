import pandas as pd
import unicodedata

def limpiar_texto(texto):
    """Normaliza texto: quita tildes, pasa a mayúsculas y quita espacios."""
    if isinstance(texto, str):
        nfkd_form = unicodedata.normalize('NFKD', texto)
        sin_tildes = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        return sin_tildes.strip().upper()
    return texto

# Rutas de tus archivos originales
ruta_matriz = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\matriz_colegios_con_comunas.parquet"
ruta_historico = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\historico_volatilidad_riesgo.parquet"
ruta_salida = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

print("⏳ Iniciando procesamiento y limpieza de datos...")

# 1. Cargar datos
df_matriz = pd.read_parquet(ruta_matriz)
df_historico = pd.read_parquet(ruta_historico)

# 2. NORMALIZACIÓN DE LLAVES (RBD)
df_matriz['RBD'] = df_matriz['RBD'].astype(str).str.split('.').str[0]
df_historico['RBD'] = df_historico['RBD'].astype(str).str.split('.').str[0]

# 3. LIMPIEZA DE COLUMNAS DE TEXTO (Protección contra tildes y eñes)
cols_texto = ['REGION', 'PROVINCIA', 'COMUNA']
for col in cols_texto:
    if col in df_matriz.columns:
        df_matriz[col] = df_matriz[col].apply(limpiar_texto)

# 4. FUSIÓN LIMPIA
df_final = pd.merge(
    df_matriz, 
    df_historico[['RBD', 'Anio', 'Volatilidad_Rendimiento']], 
    on=['RBD', 'Anio'], 
    how='left'
)

# 5. TRATAMIENTO DE NULOS (CASOS NO GRAVES)
# Asignamos 0.0 a los colegios que no registraron volatilidad grave en ese año
df_final['Volatilidad_Rendimiento'] = df_final['Volatilidad_Rendimiento'].fillna(0.0)

# 6. EXPORTACIÓN
df_final.to_parquet(ruta_salida, index=False)
print(f"✅ ÉXITO: Dataset fusionado y sin nulos guardado en {ruta_salida}")