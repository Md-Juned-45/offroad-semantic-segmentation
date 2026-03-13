import torch, os, random
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Added missing class 600 (Flowers)
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10}
n_classes = len(value_map)  # 11

def convert_mask(mask_pil):
    arr = np.array(mask_pil).astype(np.int32)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr  # return numpy for albumentations

def get_train_augmentation(h, w):
    return A.Compose([
        A.Resize(h, w, interpolation=1),  # bilinear for image
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.5),
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.ToGray(p=0.1), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_augmentation(h, w):
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, data_id)).convert("RGB"))
        mask_pil = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask_pil)  # numpy uint8

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']       # (3, H, W) float tensor
            mask = augmented['mask'].long()  # (H, W) long tensor

        return image, mask

def calc_combined_loss(preds, targets, ce_loss_fn):
    ce_loss = ce_loss_fn(preds, targets)
    preds_softmax = F.softmax(preds, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0,3,1,2).float()
    intersection = (preds_softmax * targets_one_hot).sum(dim=(2,3))
    union = preds_softmax.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    dice_loss = 1.0 - ((2.*intersection + 1e-6) / (union + 1e-6)).mean()
    return ce_loss + dice_loss

def compute_iou(pred, target, num_classes=n_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())
    return iou_per_class  # return per-class for logging

def evaluate_metrics(model, data_loader, device):
    all_iou = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            all_iou.append(compute_iou(outputs, labels))
    model.train()

    # per-class mean IoU
    all_iou = np.array(all_iou)  # (N_batches, n_classes)
    per_class_iou = np.nanmean(all_iou, axis=0)
    mean_iou = np.nanmean(per_class_iou)

    class_names = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
                   'Ground Clutter','Flowers','Logs','Rocks','Landscape','Sky']
    print("\n  Per-class IoU:")
    for name, iou in zip(class_names, per_class_iou):
        print(f"    {name:<18}: {iou:.4f}" if not np.isnan(iou) else f"    {name:<18}: N/A (not in val set)")

    return mean_iou

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 8   # T4 can handle this at 512x512 with AMP
    w, h = 512, 512
    n_epochs = 60

    data_dir = './data/train'
    val_dir  = './data/val'

    print("Checking dataset paths...")
    if not os.path.exists(data_dir):
        print(f"WARNING: Training data directory not found at {data_dir}. Please place dataset in ./data/train and ./data/val")
        return

    trainset = MaskDataset(data_dir, transform=get_train_augmentation(h, w))
    valset   = MaskDataset(val_dir,  transform=get_val_augmentation(h, w))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(valset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    class_weights = torch.tensor([
        0.5,   # Background
        1.2,   # Trees
        1.5,   # Lush Bushes
        1.5,   # Dry Grass
        1.5,   # Dry Bushes
        2.0,   # Ground Clutter
        2.5,   # Flowers 
        2.5,   # Logs
        2.0,   # Rocks
        1.0,   # Landscape
        0.5,   # Sky
    ]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print("Loading DeepLabV3+ with MiT-B2 backbone...")
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    ).to(device)

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 5e-5},
        {'params': model.decoder.parameters(), 'lr': 2e-4},
        {'params': model.segmentation_head.parameters(), 'lr': 2e-4},
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_iou = 0.0

    print(f"\nStarting training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        epoch_loss = 0.0

        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
                outputs = F.interpolate(outputs, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss = calc_combined_loss(outputs, labels, ce_loss_fn)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        val_iou = evaluate_metrics(model, val_loader, device)
        print(f"\nEpoch {epoch+1} — Val mIoU: {val_iou:.4f} | Avg Train Loss: {epoch_loss/len(train_loader):.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"*** Best model saved! mIoU: {best_val_iou:.4f} ***")

    print(f"\nDone! Best mIoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()
