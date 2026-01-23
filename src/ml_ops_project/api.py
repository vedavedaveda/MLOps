from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

try:
    from src.ml_ops_project.model import CNN
except ModuleNotFoundError:
    from model import CNN

# MODEL_PATH = Path("outputs/2026-01-21/12-11-02/cnn_model.pth") # Adjust with chosen model path
MODEL_PATH = Path("models/cnn_model.pth")

DEVICE = torch.device("cpu")

image_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

model: CNN | None = None


class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_class: str
    probabilities: dict[str, float]


def load_model(model_path: Path) -> CNN:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path.resolve()}\n"
            "Run training first: uv run python train.py\n"
            "so cnn_model.pth is created."
        )

    loaded_model = CNN()
    state_dict = torch.load(model_path, map_location=DEVICE)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="CNN Inference API", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "API is running. Use POST /predict with an image file."}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    input_tensor = image_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected model output shape: {tuple(logits.shape)} (expected [1, 2])",
            )

        probabilities_tensor = torch.softmax(logits, dim=1)[0]
        predicted_label = int(torch.argmax(probabilities_tensor).item())

        probabilities = {
            "AI": float(probabilities_tensor[0].item()),
            "Real": float(probabilities_tensor[1].item()),
        }

    predicted_class = "AI" if predicted_label == 0 else "Real"

    return PredictionResponse(
        predicted_label=predicted_label,
        predicted_class=predicted_class,
        probabilities=probabilities,
    )
