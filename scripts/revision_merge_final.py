import pandas as pd

# Ruta de tu dataset auditado
ruta_final = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

print("🔍 INICIANDO ESCÁNER DE AUDITORÍA FINAL...\n")

try:
    df_final = pd.read_parquet(ruta_final)
    
    # --- 1. ESTRUCTURA GENERAL ---
    print("--- 1. ESTRUCTURA DEL DATASET ---")
    print(f"📊 Total de filas: {len(df_final)}")
    
    # --- 2. CHEQUEO DEL MERGE (VOLATILIDAD) ---
    print("\n--- 2. SALUD DE LA VOLATILIDAD ---")
    if 'Volatilidad_Rendimiento' in df_final.columns:
        nulos_volatilidad = df_final['Volatilidad_Rendimiento'].isna().sum()
        casos_cero = (df_final['Volatilidad_Rendimiento'] == 0.0).sum()
        casos_graves = (df_final['Volatilidad_Rendimiento'] > 0.0).sum()
        
        print(f"⚠️ Nulos restantes: {nulos_volatilidad}")
        print(f"🟢 Casos 'Normales' (Valor 0.0): {casos_cero}")
        print(f"🔴 Casos 'Graves' (Valor > 0.0): {casos_graves}")
        
        if nulos_volatilidad == 0:
            print("   ✅ ¡ÉXITO! Ya no hay valores vacíos en la matriz.")
    else:
        print("   ❌ ERROR: La columna 'Volatilidad_Rendimiento' desapareció.")

    # --- 3. REPASO DE CASOS COMPLEJOS CHILENOS ---
    print("\n--- 3. CASOS COMPLEJOS (REGIONES Y COMUNAS) ---")
    if 'REGION' in df_final.columns:
        nuble_err_count = df_final[df_final['REGION'].str.contains('ÑUBLE', na=False, case=False)]
        biobio_err_count = df_final[df_final['REGION'].str.contains('BIOBÍO|BIO BÍO', na=False, case=False)]
        print(f"   ➤ Registros con Ñ en Ñuble (Debe ser 0): {len(nuble_err_count)}")
        print(f"   ➤ Registros con tilde en Biobío (Debe ser 0): {len(biobio_err_count)}")

    print("\n✅ ESCÁNER FINALIZADO.")

except FileNotFoundError:
    print(f"❌ No se encontró el archivo en la ruta: {ruta_final}")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")