from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "knee_xray_model.pth"

# Inference transform matching training normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_cached_model: nn.Module | None = None
_cached_classes: List[str] = []


def load_xray_model() -> Tuple[nn.Module, List[str]]:
  """Loads and caches the fine-tuned ResNet18 model."""
  global _cached_model, _cached_classes
  if _cached_model is not None and len(_cached_classes) > 0:
    return _cached_model, _cached_classes

  if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

  checkpoint = torch.load(MODEL_PATH, map_location="cpu")
  raw_classes = checkpoint.get("class_names", ["Normal", "Osteoarthritis"])
  class_names: List[str] = [str(c) for c in raw_classes]
  num_classes = int(checkpoint.get("num_classes", len(class_names)))

  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  _cached_model = model
  _cached_classes = class_names
  return model, class_names


def predict_xray(image_input):
  """Takes a PIL Image or file path and returns class predictions and confidence scores."""
  model, class_names = load_xray_model()

  if not isinstance(image_input, Image.Image):
    image = Image.open(image_input).convert("RGB")
  else:
    image = image_input.convert("RGB")

  # Explicit type narrowing for Pylance
  transformed = transform(image)
  assert isinstance(transformed, torch.Tensor)
  tensor = transformed.unsqueeze(0)

  with torch.no_grad():
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    conf, pred_idx = torch.max(probabilities, dim=0)

  pred_class = class_names[int(pred_idx.item())]
  conf_score = float(conf.item() * 100.0)
  prob_dict = {
      cls: float(probabilities[i].item() * 100.0)
      for i, cls in enumerate(class_names)
  }

  return pred_class, conf_score, prob_dict