import io
import random

from locust import HttpUser, between, task
from PIL import Image


def make_test_png_bytes(width: int = 32, height: int = 32) -> bytes:
    """
    Generates a simple in-memory PNG (same idea as your API tests).
    """
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiUser(HttpUser):
    """
    Simulates users calling your FastAPI inference API.
    """
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        # Create one PNG per user and reuse it (faster + more realistic)
        self.png_bytes = make_test_png_bytes()

    @task(1)
    def get_root(self) -> None:
        self.client.get("/")

    @task(3)
    def post_predict(self) -> None:
        # Upload the image as multipart/form-data just like in real usage
        self.client.post(
            "/predict",
            files={"image": ("test.png", self.png_bytes, "image/png")},
        )
