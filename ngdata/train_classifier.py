import json
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2

PRODUCT_ROOT = Path('/home/devstar18471/product_images')
CATALOG_PATH = Path('/home/devstar18471/ngd-object-detection/artifacts/category_catalog.json')
IMG_SIZE = 300
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4
NUM_CLASSES = 356

catalog = json.loads(CATALOG_PATH.read_text())
product_code_to_cat = {}
for cat_id, record in catalog.items():
    if record.get('product_code'):
        product_code_to_cat[record['product_code']] = int(cat_id)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

class ProductDataset(Dataset):
    def __init__(self, transform):
        self.samples = []
        self.transform = transform
        for product_dir in sorted(PRODUCT_ROOT.iterdir()):
            if not product_dir.is_dir():
                continue
            code = product_dir.name
            if code not in product_code_to_cat:
                continue
            cat_id = product_code_to_cat[code]
            for img_path in product_dir.rglob('*'):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cat_id = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), cat_id

transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = ProductDataset(transform)
print(f'{len(dataset)} product images, {len(set(s[1] for s in dataset.samples))} categories')

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_loss = float('inf')
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
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), '/home/devstar18471/ngd-object-detection/artifacts/classifier_finetuned.pt')
        print('  -> Saved best model')

print('Done!')
