import pandas as pd

from sklearn.model_selection import train_test_split

from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import os

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class HAMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df['target'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['full_path']).convert('RGB')
        label = self.class_to_idx[row['target']]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    base_dir = 'archive'
    metadata = pd.read_csv(base_dir + '/HAM10000_metadata.csv')

    binary_map = {
        "nv": "benign",
        "bkl":"benign",
        "df": "benign",
        "vasc": "benign",
        "mel": "Not Benign",
        "bcc":  "Not Benign",
        "akiec":  "Not Benign"
    }

    metadata["target"] = metadata["dx"].map(binary_map)
    
    lesion_counts = metadata['lesion_id'].value_counts()
    metadata['duplicates'] = metadata['lesion_id'].map(lambda x: 'duplicated' if lesion_counts[x] > 1 else 'unduplicated')
      
    image_lookup = {}
    image_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
    for image_dir in image_dirs:
        files = os.listdir(base_dir + '/' + image_dir)
        
        for file in files:
            image_lookup[file] = base_dir + '/' + image_dir

    metadata['image_path'] = metadata['image_id'] + '.jpg'
    metadata['full_path'] = metadata['image_path'].apply(lambda x: os.path.join(image_lookup[x], x))

    sampled_metadata = metadata.groupby('lesion_id').sample(n=1, random_state=42).reset_index(drop=True)

    IMG_SIZE = (224, 224)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # Transforms for validation & test (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    train_df, temp_df = train_test_split(
        sampled_metadata, test_size=0.3, stratify=sampled_metadata['target'], random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42
    )

    train_dataset = HAMDataset(train_df, transform=train_transforms)
    val_dataset = HAMDataset(val_df, transform=val_test_transforms)
    test_dataset = HAMDataset(test_df, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pretrained ResNet101
    model = models.resnet101(pretrained=True)

    # Optionally freeze all layers (for transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer
    num_classes = len(train_dataset.classes)  # Should be 2 for HAM10000
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Unfreeze the last ResNet block + FC for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    # Move model to GPU
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100 * correct / total:.2f}%")

main()