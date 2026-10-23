from fastapi.testclient import TestClient
from housing_mlops.api import app

client=TestClient(app)
EXAMPLE={"crim":0.00632,"zn":18,"indus":2.31,"chas":0,"nox":0.538,"rm":6.575,"age":65.2,"dis":4.09,"rad":1,"tax":296,"ptratio":15.3,"lstat":4.98}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


def test_health():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ok"


def test_predict():
    response=client.post("/predict",json=EXAMPLE)
    assert response.status_code==200
    assert response.json()["median_value_thousands_usd_historical"]>0


def test_input_rejection():
    bad={**EXAMPLE,"rm":-1}
    assert client.post("/predict",json=bad).status_code==422
    assert client.get("/metrics").status_code==200
