# Cargar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar datos
ruta = "Mall_Customers.csv"
df = pd.read_csv(ruta)

# Seleccionar columnas 
cols = ["Annual Income (k$)", "Spending Score (1-100)"]
data = df[cols].copy()

# Valores faltantes 
data = data.fillna(data.median(numeric_only=True))

# Escalar datos 
scaler = StandardScaler()
X = scaler.fit_transform(data.values)

# Elegir k
inercia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inercia.append(km.inertia_)

# Graficar el codo para decidir k
plt.figure()
plt.plot(list(K), inercia, marker="o")
plt.title("Método del codo")
plt.xlabel("k")
plt.ylabel("Inercia")
plt.tight_layout()
plt.show()

# Escoge k mirando el codo 
k = 5  

# Entrenar K-Means con ese k
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

# Agregar el cluster a la tabla original
df["Cluster"] = labels

# Ver centros en escala original 
centros_escalados = kmeans.cluster_centers_
centros_original = scaler.inverse_transform(centros_escalados)

# Guardar resultados a CSV
df.to_csv("datos_con_clusters.csv", index=False)

# Mostrar scatter simple con los clusters 
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=25)
plt.scatter(centros_escalados[:, 0], centros_escalados[:, 1], marker="X", s=200)
plt.title(f"Clusters (k={k})")
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.tight_layout()
plt.show()

# Imprimir centros en escala original 
print("Centros (escala original):")
for i, c in enumerate(centros_original):
    print(f"Cluster {i}: {cols[0]}={c[0]:.2f}, {cols[1]}={c[1]:.2f}")