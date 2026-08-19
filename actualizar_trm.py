import json
import requests
import pandas as pd

def actualizar_datos_trm():
    url = "https://www.datos.gov.co/resource/32sa-213a.json?$order=vigenciasta%20DESC&$limit=30"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df['valor'] = df['valor'].astype(float)
        
        datos = {
            "actual": df.iloc[0]['valor'],
            "min": df['valor'].min(),
            "max": df['valor'].max(),
            "prom": df['valor'].mean(),
            "fecha_actualizacion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("datos_trm.json", "w") as f:
            json.dump(datos, f, indent=4)
            
        print("✅ TRM actualizada exitosamente en datos_trm.json")
    except Exception as e:
        print(f"❌ Error al actualizar TRM: {e}")

if __name__ == "__main__":
    actualizar_datos_trm()
