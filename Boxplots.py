# Analisis_Mall_Customers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# rutas
csv_path = Path("Mall_Customers.csv")
out_dir = Path("out")
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# cargar
df = pd.read_csv(csv_path)
df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
df.columns = (df.columns
              .str.strip()
              .str.replace(" ", "_")
              .str.replace("-", "_"))

# vista general
print("Forma:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nNulos por columna:\n", df.isna().sum())
print("\nDuplicados:", df.duplicated().sum())

# resumen numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
resumen = df[num_cols].describe().T
resumen.to_csv(out_dir / "resumen_numericas.csv")
print("\nResumen numéricas (guardado en out/resumen_numericas.csv):\n", resumen)

# histogramas
for c in num_cols:
    plt.figure()
    df[c].dropna().plot(kind="hist", bins=20)
    plt.title(f"Histograma de {c}")
    plt.xlabel(c); plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(fig_dir / f"hist_{c}.png", dpi=150)
    plt.show()

# boxplots
for c in num_cols:
    plt.figure()
    plt.boxplot(df[c].dropna(), labels=[c])
    plt.title(f"Boxplot de {c}")
    plt.ylabel(c)
    plt.tight_layout()
    plt.savefig(fig_dir / f"box_{c}.png", dpi=150)
    plt.show()

# conteo por categoría si existe Gender
if "Gender" in df.columns:
    conteo = df["Gender"].value_counts(dropna=False)
    conteo.to_csv(out_dir / "conteo_gender.csv")
    print("\nConteo Gender (guardado en out/conteo_gender.csv):\n", conteo)
    plt.figure()
    conteo.plot(kind="bar")
    plt.title("Conteo por Gender")
    plt.xlabel("Gender"); plt.ylabel("Conteo")
    plt.tight_layout()
    plt.savefig(fig_dir / "bar_gender.png", dpi=150)
    plt.show()

# correlación + heatmap
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    corr.to_csv(out_dir / "correlacion.csv")
    print("\nCorrelación (guardado en out/correlacion.csv):\n", corr)

    plt.figure()
    im = plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Matriz de correlación")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_correlacion.png", dpi=150)
    plt.show()

# atípicos por IQR
rows = []
for c in num_cols:
    s = df[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    out_count = int(((s < low) | (s > high)).sum())
    rows.append({"columna": c, "q1": q1, "q3": q3, "iqr": iqr,
                 "low_cap": low, "high_cap": high, "outliers": out_count})
outliers_iqr = pd.DataFrame(rows).sort_values("outliers", ascending=False)
outliers_iqr.to_csv(out_dir / "outliers_iqr.csv", index=False)
print("\nOutliers por IQR (guardado en out/outliers_iqr.csv):\n", outliers_iqr)

# dispersión ingreso vs score si existen
cand_x = None
for name in ["Annual_Income_(k$)", "Annual_Income_k", "Income"]:
    if name in df.columns:
        cand_x = name; break
cand_y = None
for name in ["Spending_Score_(1-100)", "Spending_Score", "Score"]:
    if name in df.columns:
        cand_y = name; break

if cand_x and cand_y:
    plt.figure()
    plt.scatter(df[cand_x], df[cand_y])
    plt.xlabel(cand_x); plt.ylabel(cand_y)
    plt.title(f"{cand_x} vs {cand_y}")
    plt.tight_layout()
    plt.savefig(fig_dir / "scatter_ingreso_vs_score.png", dpi=150)
    plt.show()