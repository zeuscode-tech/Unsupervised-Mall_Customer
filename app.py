import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

DATASET_FILE = "Mall_Customers.csv"
FEATURE_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]

dataset_df = pd.read_csv(DATASET_FILE)

CLUSTER_NAMES = {
    0: "Средний класс",
    1: "VIP",
    2: "Импульсивные",
    3: "Осторожные",
    4: "Экономные",
}

app = FastAPI(title="Mall Customers Clustering")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/dataset-info")
def dataset_info():
    preview_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]
    cluster_distribution = (
        pd.Series(model.predict(scaler.transform(dataset_df[FEATURE_COLUMNS])))
        .value_counts()
        .sort_index()
        .to_dict()
    )
    point_clusters = model.predict(scaler.transform(dataset_df[FEATURE_COLUMNS]))
    cluster_names = {
        str(k): CLUSTER_NAMES.get(int(k), f"Cluster {k}")
        for k in cluster_distribution.keys()
    }
    model_name = type(model).__name__
    n_clusters = getattr(model, "n_clusters", None)
    model_desc = (
        f"{model_name} (n_clusters={n_clusters})"
        if n_clusters is not None
        else model_name
    )

    return {
        "dataset_name": DATASET_FILE,
        "model": model_desc,
        "features_used": FEATURE_COLUMNS,
        "preview_columns": preview_columns,
        "first_10_rows": dataset_df[preview_columns].head(10).to_dict(orient="records"),
        "stats": {
            "rows": int(dataset_df.shape[0]),
            "columns": int(dataset_df.shape[1]),
            "missing_values": int(dataset_df.isna().sum().sum()),
            "avg_income": round(float(dataset_df["Annual Income (k$)"].mean()), 2),
            "avg_spending": round(float(dataset_df["Spending Score (1-100)"].mean()), 2),
            "min_income": float(dataset_df["Annual Income (k$)"].min()),
            "max_income": float(dataset_df["Annual Income (k$)"].max()),
            "min_spending": float(dataset_df["Spending Score (1-100)"].min()),
            "max_spending": float(dataset_df["Spending Score (1-100)"].max()),
        },
        "cluster_distribution": {str(k): int(v) for k, v in cluster_distribution.items()},
        "cluster_names": cluster_names,
        "n_clusters": int(n_clusters) if n_clusters is not None else len(cluster_distribution),
        "scatter_points": [
            {
                "income": float(row["Annual Income (k$)"]),
                "spending": float(row["Spending Score (1-100)"]),
                "cluster": int(cluster_id),
            }
            for (_, row), cluster_id in zip(dataset_df.iterrows(), point_clusters)
        ],
    }


class CustomerData(BaseModel):
    income: float
    spending: float


@app.post("/predict")
def predict_cluster(data: CustomerData):
    features = np.array([[data.income, data.spending]])
    features_scaled = scaler.transform(features)

    cluster_id = int(model.predict(features_scaled)[0])

    return {
        "cluster": cluster_id,
        "name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
    }
