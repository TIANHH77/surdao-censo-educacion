import pandas as pd

# Reemplaza con tu ruta local exacta al nuevo archivo
ruta = r"C:\PROYECTOS\Agente_Alura\data\educacion\dataset_auditoria_final.parquet"

df = pd.read_parquet(ruta)
print("Las columnas exactas de tu archivo de educación son:")
print(df.columns.tolist())