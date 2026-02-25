import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# =============================
# CONFIG
# =============================

BASE_DIR = "data"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "Segmentation")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "Color_Images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val", "Segmentation")

NUM_CLASSES = 6
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# DATASET
# =============================

CLASS_VALUES = [0, 1, 2, 3, 27, 39]

def remap_mask(mask):
    new_mask = np.zeros_like(mask)
    for idx, val in enumerate(CLASS_VALUES):
        new_mask[mask == val] = idx
    return new_mask

class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        mask = remap_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# =============================
# TRANSFORMS
# =============================

train_transform = A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_dataset = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================
# MODEL
# =============================

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(device)

# =============================
# LOSS
# =============================

ce_loss = torch.nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES)
    target_onehot = target_onehot.permute(0,3,1,2).float()

    intersection = (pred * target_onehot).sum()
    union = pred.sum() + target_onehot.sum()

    return 1 - (2. * intersection + smooth) / (union + smooth)

def compute_iou(pred, mask):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(NUM_CLASSES):
        pred_inds = (pred == cls)
        target_inds = (mask == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return torch.tensor(0.0)

    return torch.mean(torch.stack(ious))

# =============================
# TRAIN LOOP
# =============================

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_iou = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            val_iou += compute_iou(outputs, masks).item()

    val_iou /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Val mIoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        os.makedirs("runs", exist_ok=True)
        torch.save(model.state_dict(), "runs/best_model.pth")
        print("Best model saved!")

print("Training Finished")
