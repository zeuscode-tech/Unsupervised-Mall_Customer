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
    return {
        "dataset_name": DATASET_FILE,
        "model": "KMeans (n_clusters=5)",
        "features_used": FEATURE_COLUMNS,
        "preview_columns": preview_columns,
        "first_10_rows": dataset_df[preview_columns].head(10).to_dict(orient="records"),
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
        "name": CLUSTER_NAMES.get(cluster_id, "Неизвестно"),
    }
