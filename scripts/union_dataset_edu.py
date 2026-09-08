import pandas as pd
import unicodedata

def limpiar_texto(texto):
    """Normaliza texto: quita tildes, pasa a mayúsculas y quita espacios."""
    if isinstance(texto, str):
        # Normaliza Unicode (quita tildes)
        nfkd_form = unicodedata.normalize('NFKD', texto)
        sin_tildes = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        return sin_tildes.strip().upper()
    return texto

# Rutas
ruta_matriz = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\matriz_colegios_con_comunas.parquet"
ruta_historico = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\historico_volatilidad_riesgo.parquet"
ruta_salida = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

# Cargar
df_matriz = pd.read_parquet(ruta_matriz)
df_historico = pd.read_parquet(ruta_historico)

# --- 1. NORMALIZACIÓN DE LLAVES ---
# Convertimos RBD a string limpio (quitando .0 si existiera)
df_matriz['RBD'] = df_matriz['RBD'].astype(str).str.split('.').str[0]
df_historico['RBD'] = df_historico['RBD'].astype(str).str.split('.').str[0]

# --- 2. LIMPIEZA DE COLUMNAS DE TEXTO (Para evitar errores con ñ, tildes, espacios) ---
cols_texto = ['REGION', 'PROVINCIA', 'COMUNA']
for col in cols_texto:
    if col in df_matriz.columns:
        df_matriz[col] = df_matriz[col].apply(limpiar_texto)

# --- 3. FUSIÓN LIMPIA ---
df_final = pd.merge(
    df_matriz, 
    df_historico[['RBD', 'Anio', 'Volatilidad_Rendimiento']], 
    on=['RBD', 'Anio'], 
    how='left'
)

# --- 4. EXPORTACIÓN ---
df_final.to_parquet(ruta_salida, index=False)
print(f"✅ Auditoría: Dataset limpiado y fusionado con éxito en {ruta_salida}")