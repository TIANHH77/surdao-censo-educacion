import os
import pandas as pd

def cargar_datamart_completo():
    dfs = {}
    # Tu ruta maestra (usamos la 'r' antes de las comillas para que Windows no llore con los slashes)
    ruta_carpeta = r"C:\SURDAO_OS\SURDAO_CENTRO_MANDO\SURDAO.ORG\DATA\indicadores_limpios"
    
    print(f"🚀 Iniciando carga masiva desde: {ruta_carpeta}")
    
    if not os.path.exists(ruta_carpeta):
        print("❌ ¡Alerta! La carpeta no existe. Revisa la ruta.")
        return dfs

    # Escaneamos todos los archivos de la carpeta
    archivos = os.listdir(ruta_carpeta)
    archivos_parquet = [f for f in archivos if f.endswith('.parquet')]
    
    for archivo in archivos_parquet:
        # Armamos la ruta completa
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        
        # Creamos un nombre amigable (Ej: "P1_2 Población de 5 años o más...")
        nombre_tabla = archivo.replace('.parquet', '').replace('_', ' ')
        
        try:
            # Cargamos el dataframe y matamos duplicados
            df = pd.read_parquet(ruta_completa)
            df = df.drop_duplicates()
            
            # Lo inyectamos al diccionario
            dfs[nombre_tabla] = df
            print(f"✅ Cargada: {nombre_tabla} ({df.shape[0]:,} filas)")
            
        except Exception as e:
            print(f"⚠️ Error cargando {archivo}: {e}")
            
    print(f"🔥 Carga finalizada: {len(dfs)} tablas activas en memoria.")
    return dfs

# Para usarlo, simplemente llamas a la función:
# diccionario_dfs = cargar_datamart_completo()