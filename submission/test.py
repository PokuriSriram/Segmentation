import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "data"
TEST_IMG_DIR = os.path.join(BASE_DIR, "testImages", "Color_Images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "testImages", "Segmentation")

NUM_CLASSES = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

test_dataset = OffroadDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)

model.load_state_dict(torch.load("runs/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

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

    return torch.mean(torch.stack(ious))

total_iou = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        total_iou += compute_iou(outputs, masks).item()

print("Test mIoU:", total_iou / len(test_loader))
