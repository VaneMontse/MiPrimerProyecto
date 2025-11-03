# Proyecto Final 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Cargar los datos
ruta = "Mall_Customers.csv"
datos = pd.read_csv(ruta)

# Ver cuántos datos hay y qué columnas tiene
print("Cantidad de filas y columnas:", datos.shape)
print("\nPrimeras filas del archivo:")
print(datos.head())

# Saber el tipo de variable de cada columna
print("\nTipos de datos:")
print(datos.dtypes)

# Ver resumen estadístico
print("\nResumen estadístico:")
print(datos.describe())

# Calcular media, mediana y desviación estándar
print("\nMedias:")
print(datos.mean(numeric_only=True))
print("\nMedianas:")
print(datos.median(numeric_only=True))
print("\nDesviación estándar:")
print(datos.std(numeric_only=True))

# Gráficas 
# Histograma de la edad
plt.hist(datos["Age"], bins=10, color="lightblue", edgecolor="black")
plt.title("Histograma de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot del ingreso anual y gasto
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(y=datos["Annual Income (k$)"], color="pink")
plt.title("Ingreso anual")

plt.subplot(1,2,2)
sns.boxplot(y=datos["Spending Score (1-100)"], color="lightgreen")
plt.title("Puntaje de gasto")
plt.show()

# Mapa de calor (correlaciones)
plt.figure(figsize=(6,4))
sns.heatmap(datos.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Mapa de calor de correlaciones")
plt.show()

# Clustering K-Means
# Seleccionar las columnas numéricas que usaremos
x = datos[["Annual Income (k$)", "Spending Score (1-100)"]]

# Método del codo para ver el mejor k
inercia = []
for k in range(1,11):
    modelo = KMeans(n_clusters=k, random_state=0)
    modelo.fit(x)
    inercia.append(modelo.inertia_)

plt.plot(range(1,11), inercia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.show()

# Usar el k que se ve con el codo (por ejemplo 5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
datos["Cluster"] = kmeans.fit_predict(x)

# Ver los centros de cada grupo
print("\nCentros de los clusters:")
print(kmeans.cluster_centers_)

# raficar los clusters
plt.scatter(x["Annual Income (k$)"], x["Spending Score (1-100)"], 
            c=datos["Cluster"], cmap="rainbow")
plt.title("Clientes agrupados por Clustering")
plt.xlabel("Ingreso anual (k$)")
plt.ylabel("Puntaje de gasto (1-100)")
plt.show()

# Conclusiones 
print("\nConclusiones:")
print("→ Los grupos se formaron según ingreso y gasto.")
print("→ Podemos ver que hay clientes con alto ingreso y bajo gasto, y viceversa.")
print("→ Los centros representan el promedio de cada grupo.")
print("→ Esto sirve para identificar tipos de clientes: ahorradores, gastadores, etc.")