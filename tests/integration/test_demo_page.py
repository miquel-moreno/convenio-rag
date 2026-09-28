from fastapi.testclient import TestClient


def test_demo_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Pregunta a tu convenio colectivo" in response.text
    assert 'fetch("/ask"' in response.text
    assert "BOE-A-2025-7766" in response.text and "BOE-A-2022-479" in response.text


def test_demo_page_is_not_part_of_the_api_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert "/ask" in paths
