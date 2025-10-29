# Análisis del dataset de Spotify
# Calcula media, mediana y desviación estándar (solo de columnas numéricas)
# También analiza los datos cualitativos más importantes

import pandas as pd

# Leer el archivo 
datos = pd.read_csv("spotify.csv", encoding="latin-1")
if "Unnamed: 0" in datos.columns:
    datos = datos.drop(columns=["Unnamed: 0"])

# ---ANÁLISIS CUANTITATIVO ---
media = datos.mean(numeric_only=True)
mediana = datos.median(numeric_only=True)
desviacion = datos.std(numeric_only=True)

print("-MEDIA-")
for col, val in media.items():
    print(f"{col:20} {val:.6f}")

print("\n-MEDIANA-")
for col, val in mediana.items():
    print(f"{col:20} {val:.6f}")

print("\n-DESVIACIÓN ESTÁNDAR-")
for col, val in desviacion.items():
    print(f"{col:20} {val:.6f}")

# --- ANÁLISIS CUALITATIVO ---
print("\n-ANÁLISIS CUALITATIVO-")

# Género musical más común
genero_mas_comun = datos["track_genre"].mode()[0]
print(f"Género musical más común: {genero_mas_comun}")
# Artista con más canciones
artista_top = datos["artists"].mode()[0]
print(f"Artista con más canciones: {artista_top}")
# Canciones explícitas
if "explicit" in datos.columns:
    total_explicit = datos["explicit"].value_counts()
    total = total_explicit.sum()
    no_explicitas = total_explicit.get(False, 0)
    explicitas = total_explicit.get(True, 0)
    porcentaje_exp = (explicitas / total) * 100
    porcentaje_noexp = (no_explicitas / total) * 100
    print(f"\nCanciones explícitas: {explicitas} ({porcentaje_exp:.2f}%)")
    print(f"Canciones no explícitas: {no_explicitas} ({porcentaje_noexp:.2f}%)")

# --- GUARDAR RESULTADOS EN ARCHIVO ---
with open("Dataset Spotify.txt", "w", encoding="utf-8") as f:
    f.write("-MEDIA-\n")
    for col, val in media.items():
        f.write(f"{col:20} {val:.6f}\n")

    f.write("\n-MEDIANA-\n")
    for col, val in mediana.items():
        f.write(f"{col:20} {val:.6f}\n")

    f.write("\n-DESVIACIÓN ESTÁNDAR-\n")
    for col, val in desviacion.items():
        f.write(f"{col:20} {val:.6f}\n")

    f.write("\n-ANÁLISIS CUALITATIVO-\n")
    f.write(f"Género musical más común: {genero_mas_comun}\n")
    f.write(f"Artista con más canciones: {artista_top}\n")
    f.write(f"Canciones explícitas: {explicitas} ({porcentaje_exp:.2f}%)\n")
    f.write(f"Canciones no explícitas: {no_explicitas} ({porcentaje_noexp:.2f}%)\n")

print("\nAnálisis completo ")
print("Los resultados se guardaron en 'resultados_spotify.txt'")