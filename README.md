# Unsupervised Mall Customer

This project applies unsupervised learning to the Kaggle dataset `Mall_Customers.csv`.
The model groups customers into behavior-based segments using KMeans clustering.

## Unsupervised Learning

Unsupervised learning is a machine learning approach where the training data does not contain target labels.
The algorithm discovers hidden patterns and structure in the data automatically.
In this project, KMeans identifies customer groups based on annual income and spending score.

## Project Structure

- `app.py` - FastAPI backend and web page hosting.
- `train.py` - model training script.
- `terminal_test.py` - terminal input testing script for prediction.
- `Mall_Customers.csv` - dataset.
- `model.pkl`, `scaler.pkl` - saved trained artifacts.
- `static/index.html` - web page with project info and predictor UI.
- `static/cluster-figure.svg` - figure used in the web page.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Model

```powershell
python train.py
```

## Run Web App

```powershell
uvicorn app:app --reload
```

Open: `http://127.0.0.1:8000`

## Test Model From Terminal Input

Make sure the FastAPI server is running, then run:

```powershell
python terminal_test.py
```

You will be prompted for:
- annual income (k$)
- spending score (1-100)

The script sends input data to `/predict` and prints the cluster result.

## API Endpoints

- `GET /` - web page.
- `GET /dataset-info` - dataset name, model info, feature list, and first 10 rows.
- `POST /predict` - cluster prediction by `income` and `spending`.
