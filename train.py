import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=5, random_state=42, n_init=10)
model.fit(X_scaled)

df["cluster"] = model.labels_
print(df[["Annual Income (k$)", "Spending Score (1-100)", "cluster"]].head(10))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nМодель и scaler сохранены!")
print("Распределение по кластерам:")
print(df["cluster"].value_counts().sort_index())
