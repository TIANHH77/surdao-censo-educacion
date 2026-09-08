import pandas as pd

ruta_final = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

df = pd.read_parquet(ruta_final)

# Buscar la columna que contenga el nombre del colegio
col_nombre = [col for col in df.columns if 'NOM' in col.upper() or 'NOMB' in col.upper() or 'ESTAB' in col.upper()]

# Filtrar los que no tienen comuna
fantasmas = df[df['COMUNA'].isna()]

# Seleccionar qué mostrar
cols_to_show = ['RBD', 'Anio']
if col_nombre:
    cols_to_show.extend(col_nombre)
if 'Total_Alumnos' in df.columns:
    cols_to_show.append('Total_Alumnos')

print("\n👻 IDENTIDAD DE LOS FANTASMAS (Un registro por colegio):")
# Mostramos solo un registro por RBD para no repetir
print(fantasmas[cols_to_show].drop_duplicates(subset=['RBD']).to_string(index=False))