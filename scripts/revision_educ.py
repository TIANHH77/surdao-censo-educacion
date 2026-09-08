import pandas as pd

ruta_final = r"C:\PROYECTOS\Agente_Alura\SURDAO_DATA_LAKE\educacion\dataset_auditoria_final.parquet"

print("🏥 INICIANDO ESCÁNER DE SALUD INTEGRAL DEL DATASET...\n")

try:
    df = pd.read_parquet(ruta_final)
    
    # --- 1. REVISIÓN DE VALORES NULOS POR COLUMNA ---
    print("--- 1. MAPA DE VALORES NULOS ---")
    nulos_totales = df.isna().sum()
    columnas_con_nulos = nulos_totales[nulos_totales > 0]
    
    if columnas_con_nulos.empty:
        print("   ✅ ¡Excelente! NO hay absolutamente ningún valor nulo en todo el dataset.")
    else:
        print("   ⚠️ Atención, se encontraron nulos en las siguientes columnas:")
        for col, cantidad in columnas_con_nulos.items():
            porcentaje = round((cantidad / len(df)) * 100, 2)
            print(f"      - {col}: {cantidad} nulos ({porcentaje}%)")

    # --- 2. CHEQUEO DE LLAVES DUPLICADAS (RBD + Anio) ---
    print("\n--- 2. INTEGRIDAD DE FILAS (DUPLICADOS) ---")
    # Un colegio no debería tener dos registros para el mismo año
    if 'RBD' in df.columns and 'Anio' in df.columns:
        duplicados = df.duplicated(subset=['RBD', 'Anio'], keep=False).sum()
        if duplicados == 0:
            print("   ✅ Cero duplicados. Cada colegio tiene un único registro por año.")
        else:
            print(f"   ❌ ALERTA: Se encontraron {duplicados} filas con el mismo RBD y Año.")

    # --- 3. TIPOS DE DATOS ---
    print("\n--- 3. VERIFICACIÓN DE TIPOS DE DATOS ---")
    print("   Asegúrate de que las columnas numéricas sean int/float y los textos sean object/string:")
    tipos_resumen = df.dtypes.value_counts()
    for tipo, cantidad in tipos_resumen.items():
        print(f"   ➤ {cantidad} columnas de tipo: {tipo}")
        
    # Mostrar tipos específicos de columnas clave si existen
    cols_clave = ['RBD', 'Anio', 'Volatilidad_Rendimiento']
    for col in cols_clave:
        if col in df.columns:
            print(f"      - {col} -> {df[col].dtype}")

    # --- 4. DETECCIÓN DE ANOMALÍAS BÁSICAS ---
    print("\n--- 4. ANOMALÍAS EN MÉTRICAS ---")
    if 'Total_Alumnos' in df.columns:
        alumnos_negativos = (df['Total_Alumnos'] < 0).sum()
        print(f"   ➤ Colegios con alumnos negativos (debe ser 0): {alumnos_negativos}")
        
    if 'Promedio_Anual' in df.columns: # Asumiendo notas de 1 a 7 en Chile
        notas_raras = ((df['Promedio_Anual'] < 1.0) | (df['Promedio_Anual'] > 7.0)).sum()
        print(f"   ➤ Promedios fuera de rango 1.0 - 7.0: {notas_raras}")

    print("\n✅ AUDITORÍA INTEGRAL FINALIZADA.")

except FileNotFoundError:
    print(f"❌ No se encontró el archivo en la ruta: {ruta_final}")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")