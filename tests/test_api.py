import io

from fastapi.testclient import TestClient
from PIL import Image

from ml_ops_project.api import app


def _make_test_png_bytes(width: int = 32, height: int = 32) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "API is running. Use POST /predict with an image file."
        }


def test_predict():
    with TestClient(app) as client:
        png_bytes = _make_test_png_bytes()

        response = client.post(
            "/predict",
            files={"image": ("test.png", png_bytes, "image/png")},
        )

        assert response.status_code == 200

        payload = response.json()
        assert payload["predicted_label"] in [0, 1]
        assert payload["predicted_class"] in ["AI", "Real"]
        assert "probabilities" in payload
        assert set(payload["probabilities"].keys()) == {"AI", "Real"}

        # Probabilities are floats
        assert isinstance(payload["probabilities"]["AI"], float)
        assert isinstance(payload["probabilities"]["Real"], float)

        # Softmax output should sum to ~1
        prob_sum = payload["probabilities"]["AI"] + payload["probabilities"]["Real"]
        assert abs(prob_sum - 1.0) < 1e-6
