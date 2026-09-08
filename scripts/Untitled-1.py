"""
Exporta un subconjunto de tablas (las que uses en el bot de Alura) de Parquet a CSV.
Uso: ajusta la lista TABLAS_A_EXPORTAR según lo que necesites para el desafío
y corre: python exportar_a_csv.py
"""
import os
import pandas as pd

BASE_CENSO = "data/CENSO_2024"
BASE_EDU = "data/educacion"
SALIDA = "data/csv_alura"

os.makedirs(SALIDA, exist_ok=True)

TABLAS_A_EXPORTAR = {
    "matriz_educacion": os.path.join(BASE_EDU, "dataset_auditoria_final.parquet"),
    "censo_edad_envejecimiento": os.path.join(
        BASE_CENSO, "D2_2_Población_censada_por_tramo_de_edad_e_índice_de_envejecimi.parquet"
    ),
    "censo_alfabetizacion": os.path.join(
        BASE_CENSO, "P7_10_Población_de_5_años_o_más_que_sabe_leer_o_escribir_por_gr.parquet"
    ),
}

for nombre, ruta in TABLAS_A_EXPORTAR.items():
    df = pd.read_parquet(ruta)
    destino = os.path.join(SALIDA, f"{nombre}.csv")
    df.to_csv(destino, index=False, encoding="utf-8-sig")
    print(f"OK -> {destino} ({df.shape[0]} filas, {df.shape[1]} cols)")

print("\nListo. Apunta tu app de Alura a la carpeta 'data/csv_alura' en vez de a los .parquet.")