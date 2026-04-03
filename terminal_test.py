import json
from urllib import request


CLUSTER_NAMES = {
    0: "Средний класс",
    1: "VIP",
    2: "Импульсивные",
    3: "Осторожные",
    4: "Экономные",
}


def post_predict(income: float, spending: float) -> dict:
    payload = json.dumps({"income": income, "spending": spending}).encode("utf-8")
    req = request.Request(
        url="http://127.0.0.1:8000/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)



def main() -> None:
    income = float(input("Enter annual income (k$): "))
    spending = float(input("Enter spending score (1-100): "))
    result = post_predict(income, spending)
    cluster_id = int(result["cluster"])
    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    print("Prediction:")
    print(f"Cluster: {cluster_id}")
    print(f"Name: {cluster_name}")


if __name__ == "__main__":
    main()
