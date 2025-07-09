import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Data Augmentation Functions
def rotate(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = np.random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def add_stickiness(img, kernel_size=(3, 3), iterations=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)

# random augment
def randomise(img):
    if np.random.rand() < 0.5:
        img = rotate(img)
    if np.random.rand() < 0.5:
        kernel_size = np.random.choice([1, 2, 3])
        iterations = np.random.choice([1, 2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        img = cv2.erode(img, kernel, iterations=iterations)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        size = int(np.random.uniform(50, 101))
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(small, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
    if np.random.rand() < 0.5:
        img = add_stickiness(img)
    return img

# Character Dataset
class CharDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        all_labels = sorted([l for l in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, l))])
        self.label_encoder.fit(all_labels)

        for label in all_labels:
            label_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(label_dir):
                path = os.path.join(label_dir, img_name)
                self.samples.append(path)
                self.labels.append(label)

        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"图像无法读取: {self.samples[idx]}")
            return self.__getitem__((idx + 1) % len(self.samples))
        img = randomise(img)
        img = cv2.resize(img, (32, 32))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # (1, 32, 32)
        label = self.encoded_labels[idx]
        return torch.tensor(img), torch.tensor(label)

# CNN Model
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# train loop
def train():
    dataset = CharDataset("./dataset/train")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCNN(num_classes=len(dataset.label_encoder.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    class_names = dataset.label_encoder.classes_
    char_counter = {label: 0 for label in class_names}  # 初始化字符计数器

    print("train start")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        print(f"\n=== Epoch {epoch+1}/10 ===")
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/10"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Count how many times each character appears in this batch
            decoded_labels = dataset.label_encoder.inverse_transform(labels.cpu().numpy())
            for char in decoded_labels:
                char_counter[char] += 1
            
            # Print unique characters trained in this batch
            batch_chars = " ".join(sorted(set(decoded_labels)))
            print(f"Characters in batch: {batch_chars}")

        print(f"Epoch {epoch+1} Loss: {running_loss / len(loader):.4f}")

        # Print cumulative character count
        print("Cumulative character training count:")
        for k in sorted(char_counter):
            print(f"  {k}: {char_counter[k]} samples")

    torch.save(model.state_dict(), "char_cnn.pth")
    print("Training complete. Model saved as char_cnn.pth")

if __name__ == "__main__":
    train()