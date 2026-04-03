# Unsupervised Mall Customer

Проект сегментирует клиентов торгового центра методом кластеризации KMeans.

## Состав проекта

- `app.py` — FastAPI API и выдача веб-интерфейса.
- `train.py` — обучение модели на датасете `Mall_Customers.csv`.
- `model.pkl`, `scaler.pkl` — сохраненные артефакты модели.
- `static/index.html` — веб-интерфейс.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Обучение модели

```powershell
python train.py
```

## Запуск приложения

```powershell
uvicorn app:app --reload
```

Открыть в браузере: `http://127.0.0.1:8000`

## API

- `GET /` — веб-страница.
- `GET /dataset-info` — информация о датасете и первые 10 строк.
- `POST /predict` — предсказание кластера по `income` и `spending`.
