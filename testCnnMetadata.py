import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from torchvision import models
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class HAMDataset(Dataset):
    def __init__(self, df, meta_array, transform=None):
        self.df = df.reset_index(drop=True)
        self.meta_array = meta_array
        self.transform = transform
        self.classes = sorted(self.df['target'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['full_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        meta_features = torch.tensor(self.meta_array[idx], dtype=torch.float32)
        label = self.class_to_idx[row['target']]
        return image, meta_features, label

class ResNetWithMetadata(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ResNetWithMetadata, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

        # Freeze all layers (optional)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace ResNet's final layer with identity
        self.resnet.fc = nn.Identity()

        # MLP for metadata
        self.meta_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Final classifier that combines image and metadata features
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, metadata):
        image_features = self.resnet(image)           # [batch, 2048]
        metadata_features = self.meta_fc(metadata)    # [batch, 32]
        combined = torch.cat((image_features, metadata_features), dim=1)
        output = self.classifier(combined)
        return output

def evaluate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in dataloader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            outputs = model(images, metadata)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"✅ Evaluation Accuracy: {accuracy:.2f}%")
    return all_preds, all_labels

def main():
    base_dir = 'archive'
    metadata = pd.read_csv(base_dir + '/HAM10000_metadata.csv')

    IMG_SIZE = (224, 224)
    lr = 1e-4

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
    
    metadata['age'] = metadata['age'].fillna(metadata['age'].median())
    metadata['sex'] = metadata['sex'].fillna('unknown')
    metadata['localization'] = metadata['localization'].fillna('unknown')

    scaler = StandardScaler()
    metadata['age'] = scaler.fit_transform(metadata[['age']])

    # split first then we will break up the arrays into metadata for the model and metadata to facilitate the run
    train_df, temp_df = train_test_split(
        metadata, test_size=0.3, stratify=metadata['target'], random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['target'], random_state=42
    )
        
    #meta_raw = metadata[['age', 'sex', 'localization']].copy()
    #meta_encoded = pd.get_dummies(meta_raw, columns=['sex', 'localization'])

    #meta_encoded = meta_encoded.set_index(metadata.index)



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

    # Select metadata rows for each split and convert
    train_meta_encoded = train_df[['age', 'sex', 'localization']].copy()
    train_encoded = pd.get_dummies(train_meta_encoded, columns=['sex', 'localization'])
    train_meta = torch.tensor(train_encoded.loc[train_df.index].to_numpy().astype(np.float32))
    
    val_meta_encoded = val_df[['age', 'sex', 'localization']].copy()
    val_encoded = pd.get_dummies(val_meta_encoded, columns=['sex', 'localization'])
    for index, dumCol in enumerate(train_encoded.columns):
        if dumCol not in val_encoded:
            val_encoded.insert(loc=index, column=dumCol, value=False)
            
    val_meta   = torch.tensor(val_encoded.loc[val_df.index].to_numpy().astype(np.float32))
    
    test_meta_encoded = test_df[['age', 'sex', 'localization']].copy()
    test_encoded = pd.get_dummies(test_meta_encoded, columns=['sex', 'localization'])
    for index, dumCol in enumerate(train_encoded.columns):
        if dumCol not in test_encoded:
            test_encoded.insert(loc=index, column=dumCol, value=False)

    test_meta  = torch.tensor(test_encoded.loc[test_df.index].to_numpy().astype(np.float32))

    train_dataset = HAMDataset(train_df, meta_array=train_meta, transform=train_transforms)
    val_dataset   = HAMDataset(val_df, meta_array=val_meta, transform=val_test_transforms)
    test_dataset  = HAMDataset(test_df, meta_array=test_meta, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_metadata_features = train_meta.shape[1]
    num_classes = len(train_dataset.classes)

    model = ResNetWithMetadata(num_metadata_features=num_metadata_features,
                            num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for images, metadata, labels in train_loader:
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # After training is done:
    val_preds, val_labels = evaluate_model(model, val_loader, device)

    # Or for test set
    test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print(classification_report(val_labels, val_preds, target_names=train_dataset.classes))

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

main()