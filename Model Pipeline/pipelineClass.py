import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultimodalPipeline:
    """
    A comprehensive pipeline for training and evaluating a multimodal deep learning model
    that uses both image and tabular data.

    This class handles data loading, preprocessing (scaling, encoding), data splitting,
    PyTorch Dataset and DataLoader creation, model training, validation, and final evaluation.
    """
    def __init__(self,
                 model_class,
                 csv_path: str,
                 image_base_dir: str,
                 image_col: str,
                 target_col: str,
                 numeric_cols: list,
                 categorical_cols: list,
                 epochs: int = 10,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 image_size: tuple = (224, 224),
                 save_filename: str = 'best_model.pth',
                 device: str = None):
        """
        Initializes the pipeline.

        Args:
            model_class (nn.Module): The PyTorch model class to be trained.
            csv_path (str): Path to the main CSV data file.
            image_base_dir (str): The base directory where images are stored.
            image_col (str): The name of the column in the CSV containing the relative image paths.
            target_col (str): The name of the target variable column.
            numeric_cols (list): A list of column names for numeric features.
            categorical_cols (list): A list of column names for categorical features.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for DataLoaders.
            lr (float): Learning rate for the optimizer.
            image_size (tuple): The size (height, width) to which images will be resized.
            save_filename (str): The filename for saving the best model.
            device (str, optional): The device to run on ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model and training parameters
        self.model_class = model_class
        self.epochs = epochs
        self.lr = lr
        self.save_filename = save_filename
        self.batch_size = batch_size

        # Data and feature configuration
        self.csv_path = csv_path
        self.image_base_dir = image_base_dir
        self.image_col = image_col
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # Image transformations
        self.image_size = image_size
        self.train_transforms, self.val_transforms = self._get_transforms()
        
        # Will be populated by _prepare_data
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.tabular_features = []
        self.tabular_input_dim = 0
        self.num_classes = 0
        self.class_names = []
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        # Prepare all data and loaders
        self._prepare_data()

        # Initialize model, criterion, and optimizer
        self.model = self.model_class(tabular_input_dim=self.tabular_input_dim, num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_val_f1 = 0
        self.best_model_path = None


    def _get_transforms(self):
        """Defines the image transformations for training and validation sets."""
        # ImageNet normalization values
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transforms, val_transforms


    def _prepare_data(self):
        """Loads and preprocesses the data from the CSV file."""
        print("--- Preparing Data ---")
        df = pd.read_csv(self.csv_path)

        # 1. Stratified split into train (60%), validation (20%), and test (20%)
        print("Splitting data...")
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df[self.target_col])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[self.target_col])
        
        # --- Preprocess Target Column ---
        train_df['label'] = self.label_encoder.fit_transform(train_df[self.target_col])
        val_df['label'] = self.label_encoder.transform(val_df[self.target_col])
        test_df['label'] = self.label_encoder.transform(test_df[self.target_col])
        
        self.class_names = list(self.label_encoder.classes_)
        self.num_classes = len(self.class_names)
        print(f"Found {self.num_classes} classes.")

        # --- Preprocess Tabular Features ---
        print("Preprocessing tabular features...")
        # Numeric features
        train_df[self.numeric_cols] = self.scaler.fit_transform(train_df[self.numeric_cols])
        val_df[self.numeric_cols] = self.scaler.transform(val_df[self.numeric_cols])
        test_df[self.numeric_cols] = self.scaler.transform(test_df[self.numeric_cols])
        
        # Categorical features
        self.one_hot_encoder.fit(train_df[self.categorical_cols])
        cat_encoded_cols = list(self.one_hot_encoder.get_feature_names_out(self.categorical_cols))

        def encode_and_merge(df):
            encoded_data = self.one_hot_encoder.transform(df[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=cat_encoded_cols, index=df.index)
            return pd.concat([df.drop(columns=self.categorical_cols), encoded_df], axis=1)

        train_df = encode_and_merge(train_df)
        val_df = encode_and_merge(val_df)
        test_df = encode_and_merge(test_df)

        self.tabular_features = self.numeric_cols + cat_encoded_cols
        self.tabular_input_dim = len(self.tabular_features)
        print(f"Total tabular features: {self.tabular_input_dim}")
        
        # --- Create Image Paths ---
        for d in [train_df, val_df, test_df]:
            d['img_path'] = d[self.image_col].apply(lambda x: os.path.join(self.image_base_dir, x))

        # --- Create Datasets and DataLoaders ---
        print("Creating Datasets and DataLoaders...")
        train_dataset = HousingDataset(train_df, self.tabular_features, self.train_transforms)
        val_dataset = HousingDataset(val_df, self.tabular_features, self.val_transforms)
        test_dataset = HousingDataset(test_df, self.tabular_features, self.val_transforms)

        num_workers = 0 if os.name == 'nt' else 2
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
        print("--- Data Preparation Complete ---")


    def _run_epoch(self, dataloader, is_training=True):
        """Helper function to run a single epoch of training or validation."""
        self.model.train(is_training)
        
        running_loss = 0.0
        all_preds, all_labels = [], []

        desc = "Training" if is_training else "Validating"
        loop = tqdm(dataloader, desc=desc, leave=False)
        
        for images, tabular_data, labels in loop:
            images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(is_training):
                outputs = self.model(images, tabular_data)
                loss = self.criterion(outputs, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return epoch_loss, epoch_acc, epoch_f1
    

    def _save_model(self):
        """Saves the model's state dictionary, adding a suffix if the file exists."""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        base_path = os.path.join(models_dir, self.save_filename)
        
        path_to_save = base_path
        # If file exists, add a numeric suffix
        if os.path.exists(base_path):
            base_name, ext = os.path.splitext(self.save_filename)
            suffix = 1
            while True:
                new_filename = f"{base_name}_{suffix}{ext}"
                new_path = os.path.join(models_dir, new_filename)
                if not os.path.exists(new_path):
                    path_to_save = new_path
                    break
                suffix += 1

        torch.save(self.model.state_dict(), path_to_save)
        print(f"Model saved to {path_to_save}")
        return path_to_save


    def _load_model(self, path):
        """Loads a saved model state dictionary."""
        # Re-initialize a fresh model architecture
        loaded_model = self.model_class(tabular_input_dim=self.tabular_input_dim, num_classes=self.num_classes)
        loaded_model.load_state_dict(torch.load(path, map_location=self.device))
        loaded_model.to(self.device)
        loaded_model.eval()
        print(f"Model loaded from {path} and set to evaluation mode.")
        return loaded_model


    def train(self):
        """Runs the main training and validation loop."""
        print("\n🚀 Starting Training Loop...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")
        print(f"Training for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.epochs} ---")
            
            train_loss, train_acc, train_f1 = self._run_epoch(self.train_loader, is_training=True)
            val_loss, val_acc, val_f1 = self._run_epoch(self.val_loader, is_training=False)

            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Valid -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Save the best model based on validation F1 score
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = self._save_model()
                print(f"🎉 New best model saved with F1 score: {self.best_val_f1:.4f}")
        
        print("\n✅ Training complete.")
        return self.best_model_path


    def evaluate(self):
        """Evaluates the best model on the test set and prints a report."""
        if not self.best_model_path:
            print("\n⚠️ No best model found from training. Evaluating the last state.")
            self.best_model_path = self._save_model()

        print(f"\n📊 Loading best model from '{self.best_model_path}' and evaluating on the test set...")
        final_model = self._load_model(self.best_model_path)
        
        final_model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, tabular_data, labels in self.test_loader:
                images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)
                outputs = final_model(images, tabular_data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        print("\n✅ Pipeline finished successfully!")

class HousingDataset(Dataset):
    """Custom PyTorch Dataset for loading images and tabular features."""
    def __init__(self, df, tabular_features, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tabular_features = tabular_features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Image processing
            img = Image.open(row['img_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
            
            # Tabular features
            tab_feats_vals = row[self.tabular_features].values.astype(np.float32)
            tab_feats = torch.from_numpy(tab_feats_vals)
            
            # Label
            label = torch.tensor(row['label'], dtype=torch.long)
            
            return img, tab_feats, label
            
        except Exception as e:
            # On error, return the next valid sample to avoid crashing the loader
            # print(f"Error loading data at index {idx} (path: {row['img_path']}). Skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))