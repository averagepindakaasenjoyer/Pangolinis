import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime # Import datetime for timestamping
import pickle # For saving preprocessing objects

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score,
    cohen_kappa_score, log_loss, classification_report, confusion_matrix)

from shapely.affinity import translate, scale
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shapely.wkt import loads

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
                 device: str = None,
                 useMask: bool = False):
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
            useMask (bool): Whether to include mask data in the dataset and model.
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
        self.useMask = useMask

        # Image transformations
        self.image_size = image_size
        self.train_transforms, self.val_transforms = self._get_transforms()

        # Will be populated by _prepare_data or _load_preprocessors
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.tabular_features = []
        self.tabular_input_dim = 0
        self.num_classes = 0
        self.class_names = []
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        # Preprocessor saving directory
        self.preprocessor_dir = "preprocessors"
        os.makedirs(self.preprocessor_dir, exist_ok=True)

        # Prepare all data and loaders
        self._prepare_data()

        # Initialize model
        # The model_class now only takes tabular_input_dim and num_classes
        self.model = self.model_class(
            tabular_input_dim=self.tabular_input_dim,
            num_classes=self.num_classes
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_val_f1 = 0
        self.best_model_path = None

        self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []}


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

    def _rasterize_polygon(self, geom):
        size = self.image_size[0]
        bounds = geom.bounds
        geom = translate(geom, xoff=-bounds[0], yoff=-bounds[1])
        scale_x = size / (bounds[2] - bounds[0] + 1e-8)
        scale_y = size / (bounds[3] - bounds[1] + 1e-8)
        geom = scale(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))

        img = Image.new("L", self.image_size, 0)
        draw = ImageDraw.Draw(img)
        coords = [(x, size - y) for x, y in geom.exterior.coords]
        draw.polygon(coords, outline=1, fill=1)
        return np.array(img, dtype=np.float32)


    def _prepare_data(self):
        """Loads and preprocesses the data from the CSV file."""
        print("--- Preparing Data ---")
        df = pd.read_csv(self.csv_path)

        # Ensure 'geometry_wkt' is loaded and converted to shapely objects if useMask is true
        if self.useMask:
            if 'geometry_wkt' not in df.columns:
                raise ValueError("The 'geometry_wkt' column is required for mask generation when useMask=True.")
            try:
                # Assuming geometry_wkt is already a WKT string, parse it
                df['geometry_obj'] = df['geometry_wkt'].apply(loads)
                df["mask"] = df["geometry_obj"].apply(lambda g: self._rasterize_polygon(g))
                df.drop(columns=['geometry_obj'], inplace=True)
            except Exception as e:
                print(f'Error during mask rasterization: {e}. Disabling mask usage for pipeline.')
                self.useMask = False
        else:
             print("Mask usage is disabled for the pipeline.")


        # 1. Stratified split into train (70%), validation (15%), and test (15%)
        print("Splitting data...")
        # Ensure target column exists before stratifying
        if self.target_col not in df.columns:
             raise ValueError(f"Target column '{self.target_col}' not found in the DataFrame.")

        # Filter out rows where target_col might be NaN if it's an issue
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)


        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[self.target_col])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[self.target_col])

        # --- Preprocess Target Column ---
        self.label_encoder.fit(train_df[self.target_col])
        train_df['label'] = self.label_encoder.transform(train_df[self.target_col])
        val_df['label'] = self.label_encoder.transform(val_df[self.target_col])
        test_df['label'] = self.label_encoder.transform(test_df[self.target_col])

        self.class_names = list(self.label_encoder.classes_)
        self.num_classes = len(self.class_names)
        print(f"Found {self.num_classes} classes: {self.class_names}")

        # --- Preprocess Tabular Features ---
        print("Preprocessing tabular features...")
        # Numeric features
        # Filter numeric_cols to only include those present in df
        actual_numeric_cols = [col for col in self.numeric_cols if col in df.columns]
        if not actual_numeric_cols:
            print("Warning: No numeric columns found for scaling.")
        else:
            self.scaler.fit(train_df[actual_numeric_cols])
            train_df[actual_numeric_cols] = self.scaler.transform(train_df[actual_numeric_cols])
            val_df[actual_numeric_cols] = self.scaler.transform(val_df[actual_numeric_cols])
            test_df[actual_numeric_cols] = self.scaler.transform(test_df[actual_numeric_cols])

        # Categorical features
        actual_categorical_cols = [col for col in self.categorical_cols if col in df.columns]
        if not actual_categorical_cols:
            print("Warning: No categorical columns found for one-hot encoding.")
            cat_encoded_cols = []
        else:
            # Handle unknown categories during transform by making sure they exist in training set
            for col in actual_categorical_cols:
                train_df[col] = train_df[col].astype('category')
                # For val/test, add categories observed in training to prevent errors on unseen categories
                val_df[col] = pd.Categorical(val_df[col], categories=train_df[col].cat.categories)
                test_df[col] = pd.Categorical(test_df[col], categories=train_df[col].cat.categories)

            self.one_hot_encoder.fit(train_df[actual_categorical_cols])
            cat_encoded_cols = list(self.one_hot_encoder.get_feature_names_out(actual_categorical_cols))

        def encode_and_merge(df_to_encode, encoder_obj, cols_to_encode, encoded_names):
            if not cols_to_encode: # No categorical columns to encode
                return df_to_encode.copy()
            encoded_data = encoder_obj.transform(df_to_encode[cols_to_encode])
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_names, index=df_to_encode.index)
            return pd.concat([df_to_encode.drop(columns=cols_to_encode, errors='ignore'), encoded_df], axis=1)

        train_df = encode_and_merge(train_df, self.one_hot_encoder, actual_categorical_cols, cat_encoded_cols)
        val_df = encode_and_merge(val_df, self.one_hot_encoder, actual_categorical_cols, cat_encoded_cols)
        test_df = encode_and_merge(test_df, self.one_hot_encoder, actual_categorical_cols, cat_encoded_cols)

        # Define full list of tabular features AFTER encoding
        self.tabular_features = actual_numeric_cols + cat_encoded_cols
        self.tabular_input_dim = len(self.tabular_features)
        print(f"Total tabular features: {self.tabular_input_dim}")

        # Ensure all tabular features are numeric and handle NaNs
        def clean_and_convert_features(df_to_clean, feature_list):
            cleaned_df = df_to_clean.copy()
            for feature in feature_list:
                if feature in cleaned_df.columns:
                    cleaned_df[feature] = pd.to_numeric(cleaned_df[feature], errors='coerce')
                    cleaned_df[feature] = cleaned_df[feature].fillna(0) # Fill NaNs with 0
                    cleaned_df[feature] = cleaned_df[feature].astype('float32')
            return cleaned_df

        train_df = clean_and_convert_features(train_df, self.tabular_features)
        val_df = clean_and_convert_features(val_df, self.tabular_features)
        test_df = clean_and_convert_features(test_df, self.tabular_features)

        # --- Save Preprocessing Objects ---
        try:
            with open(os.path.join(self.preprocessor_dir, "scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(os.path.join(self.preprocessor_dir, "one_hot_encoder.pkl"), 'wb') as f:
                pickle.dump(self.one_hot_encoder, f)
            with open(os.path.join(self.preprocessor_dir, "label_encoder.pkl"), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            # Also save tabular_features list for consistent column order in prediction
            with open(os.path.join(self.preprocessor_dir, "tabular_features.pkl"), 'wb') as f:
                pickle.dump(self.tabular_features, f)
            print(f"Preprocessing objects saved to {self.preprocessor_dir}/")
        except Exception as e:
            print(f"Error saving preprocessing objects: {e}")

        # --- Create Image Paths ---
        for d in [train_df, val_df, test_df]:
            d['img_path'] = d[self.image_col].apply(lambda x: os.path.join(self.image_base_dir, x) if pd.notna(x) else None)

        # --- Create Datasets and DataLoaders ---
        print("Creating Datasets and DataLoaders...")
        train_dataset = HousingDataset(train_df, self.tabular_features, self.train_transforms, include_mask=self.useMask)
        val_dataset = HousingDataset(val_df, self.tabular_features, self.val_transforms, include_mask=self.useMask)
        test_dataset = HousingDataset(test_df, self.tabular_features, self.val_transforms, include_mask=self.useMask)

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

        for batch_data in loop:
            # Flexible unpacking based on whether mask is included
            if self.useMask:
                images, masks, tabular_data, labels = batch_data
                images, masks, tabular_data, labels = images.to(self.device), masks.to(self.device), tabular_data.to(self.device), labels.to(self.device)
            else:
                images, tabular_data, labels = batch_data
                images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(is_training):
                outputs = self.model(images, masks if self.useMask else None, tabular_data)

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

    def train(self, epochs = None):
        """Runs the full training and validation loop."""

        if epochs and epochs >= 1:
            self.epochs = epochs

        print("🚀 Starting full pipeline execution...")
        print(f"Training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.epochs} ---")

            train_loss, train_acc, train_f1 = self._run_epoch(self.train_loader, is_training=True)
            val_loss, val_acc, val_f1 = self._run_epoch(self.val_loader, is_training=False)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            print(f"Epoch {epoch} Summary:")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Valid -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = self._save_model()
                print(f"🎉 New best model saved with F1 score: {self.best_val_f1:.4f}")

        print("\n✅ Training complete.")

    def evaluate(self, dataloader_type='test', 
             show_confusion_matrix=True, 
             show_training_curves=True, 
             show_feature_importance=True):
        if self.best_model_path:
            print(f"\n📊 Loading best model from '{self.best_model_path}' and evaluating on the {dataloader_type} set...")
            final_model = self.load_saved_model(self.best_model_path)
        else:
            print("\n⚠️ No best model saved during training. Evaluating current model state.")
            final_model = self.model

        final_model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        dataloader = self.test_loader if dataloader_type == 'test' else self.val_loader

        with torch.no_grad():
            for batch_data in dataloader:
                if self.useMask:
                    images, masks, tabular_data, labels = batch_data
                    images, masks, tabular_data, labels = images.to(self.device), masks.to(self.device), tabular_data.to(self.device), labels.to(self.device)
                    outputs = final_model(images, masks, tabular_data)
                else:
                    images, tabular_data, labels = batch_data
                    images, tabular_data, labels = images.to(self.device), tabular_data.to(self.device), labels.to(self.device)
                    outputs = final_model(images, None, tabular_data)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)

                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # --- Metrics ---
        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
        print(f"Precision (macro): {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
        print(f"Recall (macro): {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
        print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
        print(f"Cohen's Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")
        print(f"Log Loss: {log_loss(all_labels, all_probs):.4f}")

        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=self.class_names, zero_division=0))

        # --- Confusion Matrix ---
        if show_confusion_matrix:
            print("\n--- Confusion Matrix ---")
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

        # --- Training Curves ---
        if show_training_curves:
            epochs = np.arange(1, len(self.history['train_loss']) + 1)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.history['train_loss'], label='Train Loss')
            plt.plot(epochs, self.history['val_loss'], label='Val Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(epochs)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.history['train_acc'], label='Train Accuracy')
            plt.plot(epochs, self.history['val_acc'], label='Val Accuracy')
            plt.plot(epochs, self.history['train_f1'], label='Train F1-Score')
            plt.plot(epochs, self.history['val_f1'], label='Val F1-Score')
            plt.title('Metrics over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.xticks(epochs)
            plt.legend()

            plt.tight_layout()
            plt.show()

        if show_feature_importance:
            def get_sample_batch(loader, batch_size=32):
                for batch in loader:
                    if self.useMask:
                        return [x[:batch_size] for x in batch]
                    else:
                        return [x[:batch_size] for x in batch]

            class ModelWrapper:
                def __init__(self, model, device, model_type):
                    self.model = model
                    self.device = device
                    self.model_type = model_type

                def fit(self, X_tabular, y):
                    self.X_tabular = torch.tensor(X_tabular, dtype=torch.float32).to(self.device)

                def predict(self, X_tabular):
                    self.model.eval()
                    X_tabular = torch.tensor(X_tabular, dtype=torch.float32).to(self.device)
                    dummy_images = torch.zeros(len(X_tabular), 3, 224, 224).to(self.device)
                    dummy_masks = torch.zeros(len(X_tabular), 1, 224, 224).to(self.device)
                    with torch.no_grad():
                        if self.model_type == 'MultiMask':
                            outputs = self.model(dummy_images, dummy_masks, X_tabular)
                        else:
                            outputs = self.model(dummy_images, X_tabular)
                        preds = torch.argmax(outputs, dim=1)
                    return preds.cpu().numpy()

            print("\n🔍 Analyzing tabular feature importance...")
            sample_batch = get_sample_batch(self.test_loader, batch_size=min(100, 32))

            if self.useMask:
                sample_images, sample_masks, sample_tabular, sample_labels = sample_batch
                X_combined = (sample_images.numpy(), sample_masks.numpy(), sample_tabular.numpy())
            else:
                sample_images, sample_tabular, sample_labels = sample_batch
                X_combined = (sample_images.numpy(), sample_tabular.numpy())

            y_true = sample_labels.numpy()

            model_wrapper = ModelWrapper(final_model, self.device, 'MultiMask' if self.useMask else 'SimpleModel')
            model_wrapper.fit(X_combined[2] if self.useMask else X_combined[1], y_true)

            final_model.eval()
            final_model.zero_grad()

            criterion = nn.CrossEntropyLoss()

            if self.useMask:
                images = sample_images.to(self.device).requires_grad_(True)
                masks = sample_masks.to(self.device)
                tabular = sample_tabular.to(self.device).requires_grad_(True)
                outputs = final_model(images, masks, tabular)
            else:
                images = sample_images.to(self.device)
                tabular = sample_tabular.to(self.device).requires_grad_(True)
                outputs = final_model(images, None, tabular)

            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, preds)
            loss.backward()

            if tabular.grad is not None:
                grad_importance = torch.abs(tabular.grad).mean(dim=0).cpu().numpy()
            else:
                grad_importance = np.zeros(len(self.tabular_features))

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            feature_names = [f.replace('_', ' ').title() for f in self.tabular_features]

            ax1.barh(range(len(grad_importance)), grad_importance)
            ax1.set_yticks(range(len(feature_names)))
            ax1.set_yticklabels(feature_names, fontsize=8)
            ax1.set_xlabel('Gradient Magnitude')
            ax1.set_title('Gradient-based Feature Importance')
            ax1.grid(alpha=0.3)

            top_indices = np.argsort(grad_importance)[-10:]
            ax2.barh(range(len(top_indices)), grad_importance[top_indices])
            ax2.set_yticks(range(len(top_indices)))
            ax2.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10)
            ax2.set_xlabel('Gradient Magnitude')
            ax2.set_title('Top 10 Most Important Features')
            ax2.grid(alpha=0.3)

            ax3.hist(grad_importance, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Gradient Magnitude')
            ax3.set_ylabel('Number of Features')
            ax3.set_title('Distribution of Feature Importance')
            ax3.grid(alpha=0.3)

            top_15_indices = np.argsort(grad_importance)[-15:]
            normalized_importance = grad_importance[top_15_indices] / np.sum(grad_importance[top_15_indices])

            ax4.pie(normalized_importance, labels=[feature_names[i][:15] + '...' if len(feature_names[i]) > 15 else feature_names[i] for i in top_15_indices],
                    autopct='%1.1f%%', startangle=90)
            ax4.set_title('Top Features - Relative Importance')

            plt.tight_layout()
            plt.show()

            print("\n📈 Top Most Important Features:")
            for i, idx in enumerate(top_indices[-10:][::-1]):
                print(f"  {i+1:2d}. {feature_names[idx]}: {grad_importance[idx]:.6f}")


        print(f"\n✅ Pipeline finished successfully for {dataloader_type} evaluation!")




    def _save_model(self):
        """
        Saves the model's state dictionary. It will overwrite the existing 'best_model.pth'
        from the current training run. If other models exist in the 'models' folder,
        a new timestamped file will be created to avoid overwriting them.
        """
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        base_path = os.path.join(models_dir, self.save_filename)

        # Check for other .pth files in the directory
        existing_pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and f != self.save_filename]

        path_to_save = base_path
        if existing_pth_files:
            # If other .pth files exist, create a new timestamped file
            timestamp = datetime.now().strftime("%m-%d_%H%M")
            base_name, ext = os.path.splitext(self.save_filename)
            path_to_save = os.path.join(models_dir, f"{base_name}_{timestamp}{ext}")
            print(f"Other models found in '{models_dir}'. Saving new model to '{path_to_save}' to avoid conflict.")
        else:
            # If no other .pth files, or only 'best_model.pth' from a previous run,
            # overwrite the default 'best_model.pth'
            print(f"Saving model to '{path_to_save}' (overwriting if it exists from this run).")

        torch.save(self.model.state_dict(), path_to_save)
        print(f"Model saved to {path_to_save}")
        return path_to_save


    def load_saved_model(self, path, evaluate = True):
        """Loads a saved model state dictionary."""
        # Re-initialize a fresh model architecture
        loaded_model = self.model_class(
            tabular_input_dim=self.tabular_input_dim,
            num_classes=self.num_classes
        )
        loaded_model.load_state_dict(torch.load(path, map_location=self.device))
        loaded_model.to(self.device)
        if evaluate:
            loaded_model.eval()
            print(f"Model loaded from {path} and set to evaluation mode.")
        return loaded_model

    def _load_preprocessors(self):
        """Loads saved preprocessor objects."""
        # This method is crucial and should be present and correct.
        try:
            with open(os.path.join(self.preprocessor_dir, "scaler.pkl"), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(self.preprocessor_dir, "one_hot_encoder.pkl"), 'rb') as f:
                self.one_hot_encoder = pickle.load(f)
            with open(os.path.join(self.preprocessor_dir, "label_encoder.pkl"), 'rb') as f:
                self.label_encoder = pickle.load(f)
                self.class_names = list(self.label_encoder.classes_)
                self.num_classes = len(self.class_names)
            with open(os.path.join(self.preprocessor_dir, "tabular_features.pkl"), 'rb') as f:
                self.tabular_features = pickle.load(f)
            self.tabular_input_dim = len(self.tabular_features)
            print("Preprocessing objects loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError("Preprocessor files not found. Please ensure the pipeline has been trained or preprocessors saved.")
        except Exception as e:
            print(f"Error loading preprocessors: {e}")
            raise


    def classify(self, input_csv_name: str = 'input.csv', input_image_dir: str = 'images', thresh_hold=0.5):
        """
        Classifies one or more unlabeled data entries from a CSV file and their corresponding images.

        Args:
            input_csv_name (str): The name of the CSV file containing data entries (in './input/').
            input_image_dir (str): Directory within './input/' where the images are located.
            thresh_hold (float): Certainty threshold below which the result is labeled 'Uncertain'.

        Returns:
            List[Tuple[str, List[float], int]]: A list of predictions, each being a tuple of:
                                                (predicted class name, probability list, predicted label index).
        """
        print("\n--- Starting Classification of Data Entries ---")
        self.thresh_hold = thresh_hold

        if not hasattr(self, 'scaler') or self.scaler is None:
            self._load_preprocessors()

        input_csv_path = os.path.join('..', 'input', input_csv_name)
        predictions = []

        try:
            df = pd.read_csv(input_csv_path)

            for _, row in df.iterrows():
                try:
                    img_filename = row[self.image_col]
                    img_path = os.path.join('..', 'input', input_image_dir, img_filename)

                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found at {img_path}")

                    img = Image.open(img_path).convert('RGB')
                    img = self.val_transforms(img)
                    img = img.unsqueeze(0)

                    mask = None
                    if self.useMask:
                        if 'geometry_wkt' in row and pd.notna(row['geometry_wkt']):
                            try:
                                geom_obj = loads(str(row['geometry_wkt']).strip())
                                mask_data = self._rasterize_polygon(geom_obj)
                                mask = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            except Exception as e:
                                print(f"Warning: Mask generation failed for entry with image {img_filename}: {e}")
                                mask = None
                        else:
                            print(f"Warning: 'geometry_wkt' missing for entry with image {img_filename}. Skipping mask.")

                    row_df = pd.DataFrame([row])

                    actual_numeric_cols = [col for col in self.numeric_cols if col in row_df.columns]
                    if actual_numeric_cols:
                        row_df[actual_numeric_cols] = self.scaler.transform(row_df[actual_numeric_cols])

                    actual_categorical_cols = [col for col in self.categorical_cols if col in row_df.columns]
                    if actual_categorical_cols:
                        for col in actual_categorical_cols:
                            if col in self.one_hot_encoder.feature_names_in_:
                                encoder_categories = self.one_hot_encoder.categories_[self.one_hot_encoder.feature_names_in_.tolist().index(col)]
                                row_df[col] = pd.Categorical(row_df[col], categories=encoder_categories)
                            else:
                                row_df[col] = row_df[col].astype('category')

                        encoded = self.one_hot_encoder.transform(row_df[actual_categorical_cols])
                        encoded_cols = list(self.one_hot_encoder.get_feature_names_out(actual_categorical_cols))
                        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=row_df.index)
                        row_df = pd.concat([row_df.drop(columns=actual_categorical_cols, errors='ignore'), encoded_df], axis=1)

                    for feature in self.tabular_features:
                        if feature in row_df.columns:
                            row_df[feature] = pd.to_numeric(row_df[feature], errors='coerce').fillna(0).astype('float32')
                        else:
                            row_df[feature] = 0.0

                    tabular_data = torch.tensor(row_df[self.tabular_features].values.astype(np.float32), dtype=torch.float32)

                    self.model.eval()
                    with torch.no_grad():
                        img = img.to(self.device)
                        if mask is not None:
                            mask = mask.to(self.device)
                        tabular_data = tabular_data.to(self.device)

                        outputs = self.model(img, mask if self.useMask else None, tabular_data)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_label_idx = torch.argmax(probabilities, dim=1).item()

                    certainty = probabilities.tolist()[0][predicted_label_idx]
                    if certainty < self.thresh_hold:
                        predicted_class_name = 'Uncertain'
                    else:
                        predicted_class_name = self.label_encoder.inverse_transform([predicted_label_idx])[0]

                    print(f"✅ Entry with image '{img_filename}' predicted as: {predicted_class_name}")
                    predictions.append((predicted_class_name, probabilities.tolist()[0], predicted_label_idx))

                except Exception as entry_error:
                    print(f"Error processing row: {entry_error}")
                    predictions.append(("Error", [], -1))

            return predictions

        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the 'input' folder and input files are correctly placed.")
            return []
        except Exception as e:
            print(f"Unexpected error during classification: {e}")
            return []



class HousingDataset(Dataset):
    """Custom PyTorch Dataset for loading images and tabular features."""
    def __init__(self, df, tabular_features, transform=None, include_mask=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tabular_features = tabular_features
        self.include_mask = include_mask

        self._remove_missing()

    def __len__(self):
        return len(self.df)

    def _remove_missing(self):
        initial_len = len(self.df)
        # Check if 'img_path' column exists
        if 'img_path' not in self.df.columns:
            print("Warning: 'img_path' column not found in DataFrame. Cannot check for missing images.")
            return

        img_paths = self.df['img_path'].dropna().tolist() # Get non-NA paths
        existing_paths = [path for path in img_paths if os.path.exists(path)]

        # Filter the DataFrame to keep only rows with existing image paths
        # This approach avoids iterating row by row with os.path.exists for performance
        if len(existing_paths) < len(img_paths):
            self.df = self.df[self.df['img_path'].isin(existing_paths)].reset_index(drop=True)
            print(f"Dropped {initial_len - len(self.df)} rows due to missing image paths.")
        # If the df initially had NaNs in 'img_path', those would have been dropped by dropna above.
        # So we also need to account for rows that had img_path but it was None/NaN.
        # The initial_len accounts for all rows including those with missing paths.

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get('img_path', None)

        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                # Default conversion if no transform specified, ensure it's a tensor
                img = transforms.ToTensor()(img) # Basic ToTensor for consistency

            tabular_data = torch.tensor(row[self.tabular_features].values.astype(np.float32), dtype=torch.float32)
            label = torch.tensor(row['label'], dtype=torch.long)

            if self.include_mask:
                # Ensure mask is a NumPy array before converting to tensor if not already
                mask_data = row['mask']
                if not isinstance(mask_data, np.ndarray):
                    mask_data = np.array(mask_data)
                # Reshape mask to [1, H, W] for single channel
                mask = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
                return img, mask, tabular_data, label
            else:
                # Return dummy mask if not included for consistent model forward signature
                dummy_mask = torch.zeros(1, *self.transform.transforms[0].size, dtype=torch.float32) if self.transform else torch.zeros(1, 224, 224, dtype=torch.float32)
                return img, dummy_mask, tabular_data, label # Always return a mask (dummy if not actual)
        except Exception as e:
            print(f"Error loading data at index {idx} (path: {img_path}). Error: {e}")
            # Instead of recursively calling, which can lead to infinite loops if many bad rows,
            # we can skip this item or raise a more specific error, or return a default/empty.
            # For robustness in DataLoader, dropping during _remove_missing is better.
            # If still failing, it implies an unhandled issue.
            raise RuntimeError(f"Critical error loading data at index {idx}. Check data integrity. Original error: {e}")