from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "xray"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = MODEL_DIR / "knee_xray_model.pth"

# Resolve validation directory (handles 'Valid', 'valid', or 'val')
train_dir = DATA_DIR / "train"
if (DATA_DIR / "Valid").exists():
  val_dir = DATA_DIR / "Valid"
elif (DATA_DIR / "valid").exists():
  val_dir = DATA_DIR / "valid"
elif (DATA_DIR / "val").exists():
  val_dir = DATA_DIR / "val"
else:
  val_dir = DATA_DIR / "test"


def main():
  # Select GPU if NVIDIA CUDA is available, else fallback to CPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

  # Preprocessing & standard ImageNet normalization
  train_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  val_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  print(f"Loading train set from: {train_dir}")
  print(f"Loading validation set from: {val_dir}")

  # Auto-discovers classes from subfolder names
  train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
  val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)

  # num_workers=0 avoids Windows child-process bootstrap errors
  train_loader = DataLoader(
      train_ds, batch_size=32, shuffle=True, num_workers=0
  )
  val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

  class_names = train_ds.classes
  num_classes = len(class_names)
  print(f"\nDetected {num_classes} classes: {class_names}")
  print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

  # Transfer Learning backbone: Pretrained ResNet18
  print("\nInitializing ResNet18 backbone...")
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  epochs = 5
  best_val_acc = 0.0

  print("\n" + "=" * 45)
  print("STARTING X-RAY VISION MODEL TRAINING")
  print("=" * 45)

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      _, preds = torch.max(outputs, 1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)

    epoch_train_acc = correct / total if total > 0 else 0
    epoch_loss = running_loss / total if total > 0 else 0

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

    epoch_val_acc = val_correct / val_total if val_total > 0 else 0

    print(
        f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_train_acc * 100:.2f}% | Val Acc:"
        f" {epoch_val_acc * 100:.2f}%"
    )

    if epoch_val_acc > best_val_acc:
      best_val_acc = epoch_val_acc

  # Save trained weights & class mappings
  torch.save(
      {
          "model_state_dict": model.state_dict(),
          "class_names": class_names,
          "num_classes": num_classes,
          "best_val_acc": best_val_acc,
      },
      SAVE_PATH,
  )

  print("\n" + "=" * 45)
  print("X-RAY MODEL SAVED SUCCESSFULLY")
  print("=" * 45)
  print(f"File: {SAVE_PATH}")
  print(f"Top Validation Accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
  main()