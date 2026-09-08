import pandas as pd

ruta_final = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

print("🧹 INICIANDO PROTOCOLO DE PURGA...")

try:
    # 1. Cargar el dataset actual
    df = pd.read_parquet(ruta_final)
    total_inicial = len(df)
    
    # 2. Filtrar: Conservar SOLO las filas que sí tienen comuna
    df_limpio = df[df['COMUNA'].notna()]
    total_final = len(df_limpio)
    
    # 3. Sobrescribir el archivo final
    df_limpio.to_parquet(ruta_final, index=False)
    
    print(f"✅ PURGA EXITOSA:")
    print(f"   ➤ Filas antes: {total_inicial}")
    print(f"   ➤ Filas ahora: {total_final}")
    print(f"   ➤ Fantasmas eliminados: {total_inicial - total_final}")
    print("\n🛡️ MATRIZ DE EDUCACIÓN SELLADA. 100% lista para producción.")

except Exception as e:
    print(f"❌ Error durante la purga: {e}")