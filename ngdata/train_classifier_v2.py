"""Train EfficientNet-B3 on real crops from training images."""
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2

CROPS_DIR = Path('/home/devstar18471/crops')
OUT_MODEL = Path('/home/devstar18471/ngd-object-detection/artifacts/classifier_realcrops.pt')
IMG_SIZE = 300
EPOCHS = 80
BATCH_SIZE = 64
LR = 3e-4
NUM_CLASSES = 356

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class CropDataset(Dataset):
    def __init__(self, transform):
        self.samples = []
        self.transform = transform
        for cat_dir in sorted(CROPS_DIR.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_id = int(cat_dir.name)
            for img_path in cat_dir.rglob('*'):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, cat_id))
        print(f'Dataset: {len(self.samples)} crops, {len(set(s[1] for s in self.samples))} categories')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cat_id = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), cat_id
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), cat_id

transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = CropDataset(transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    scheduler.step()
    acc = correct / total
    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.4f}')
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), str(OUT_MODEL))
        print(f'  -> Saved best (acc={acc:.4f})')

print(f'Done! Best acc: {best_acc:.4f}')
