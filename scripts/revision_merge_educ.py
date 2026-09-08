import pandas as pd

# Rutas originales
ruta_matriz = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\matriz_colegios_con_comunas.parquet"
ruta_historico = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\historico_volatilidad_riesgo.parquet"

df_matriz = pd.read_parquet(ruta_matriz)
df_historico = pd.read_parquet(ruta_historico)

print("\n🔍 --- DIAGNÓSTICO DE LLAVES DE CRUCE ---")

# 1. Revisar los Años
print("\n📅 AÑOS DISPONIBLES:")
print(f"Años en Matriz principal: {df_matriz['Anio'].unique()} (Tipo: {df_matriz['Anio'].dtype})")
print(f"Años en Histórico: {df_historico['Anio'].unique()} (Tipo: {df_historico['Anio'].dtype})")

# 2. Revisar si los colegios (RBD) realmente existen en ambos lados
rbd_matriz = set(df_matriz['RBD'].astype(str).str.split('.').str[0].unique())
rbd_historico = set(df_historico['RBD'].astype(str).str.split('.').str[0].unique())
interseccion_rbd = rbd_matriz.intersection(rbd_historico)

print("\n🏫 COINCIDENCIA DE COLEGIOS (RBD):")
print(f"Total RBDs únicos en Matriz: {len(rbd_matriz)}")
print(f"Total RBDs únicos en Histórico: {len(rbd_historico)}")
print(f"RBDs que existen en AMBOS archivos: {len(interseccion_rbd)}")