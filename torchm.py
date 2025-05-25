import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

class CustomCNN(nn.Module):
    """Optimized Custom CNN model for brain tumor classification."""
    def __init__(self, num_classes, input_channels=3):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # Reduced filters for speed
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),  # Reduced dropout rate

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),  # Adjusted for input_size=224 and pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class BrainTumorTrainer:
    """Class to handle training of a custom CNN model for brain tumor classification."""
    
    def __init__(self, args):
        """Initialize trainer with command-line arguments."""
        self.data_dir = os.path.join(args.data_dir, "training")  # Point directly to training folder
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.input_size = args.input_size
        self.learning_rate = args.learning_rate
        self.patience = args.patience
        self.fine_tune_epoch = args.fine_tune_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1s = []
        self.val_f1s = []
        self.class_names = None
    
    @staticmethod
    def _parse_arguments():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a custom CNN model for brain tumor classification.")
        parser.add_argument("--data_dir", type=str, default="brain_cancer", help="Path to the dataset directory containing 'training' folder")
        parser.add_argument("--model_name", type=str, default="Zainab", help="Name for saving the model")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")  # Increased for speed
        parser.add_argument("--num_epochs", type=int, default=20, help="Maximum number of training epochs")
        parser.add_argument("--input_size", type=int, default=224, help="Input image size (width and height)")
        parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate for the optimizer")
        parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement")
        parser.add_argument("--fine_tune_epoch", type=int, default=15, help="Epoch to adjust learning rate for fine-tuning")
        return parser.parse_args()

    def _get_data_loaders(self):
        """Create data loaders for training and validation with simplified augmentation."""
        data_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Reduced rotation range
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])

        dataset = datasets.ImageFolder(self.data_dir, transform=data_transforms)
        self.class_names = dataset.classes
        self.num_classes = len(self.class_names)  # Dynamically set num_classes
        print(f"Dataset size: {len(dataset)}, Classes: {self.class_names}")

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        labels = [dataset.targets[i] for i in train_indices]
        class_counts = Counter(labels)
        print("Training class distribution:", class_counts)

        total_samples = len(train_indices)
        class_weights = []
        for i in range(self.num_classes):
            count = class_counts.get(i, 0)
            if count == 0:
                print(f"Warning: No samples for class {i}. Skipping this class.")
                class_weights.append(0.0)  # Zero weight for missing classes
            else:
                # Proportional weight: inversely proportional to class frequency
                class_weights.append(total_samples / (self.num_classes * count))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print("Class weights:", class_weights)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler, 
            num_workers=8,  # Increased for speed
            pin_memory=True if self.device.type == "cuda" else False
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            sampler=val_sampler, 
            num_workers=8, 
            pin_memory=True if self.device.type == "cuda" else False
        )
        return train_loader, val_loader, class_weights

    def _create_model(self, fine_tune=False):
        """Create and configure custom CNN model."""
        model = CustomCNN(num_classes=self.num_classes, input_channels=3).to(self.device)
        return model

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train the model for one epoch and compute metrics."""
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        return epoch_loss, accuracy, precision, recall, f1

    def _validate_epoch(self, model, val_loader, criterion):
        """Validate the model for one epoch and compute metrics."""
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        return epoch_val_loss, accuracy, precision, recall, f1, all_preds, all_labels

    def _plot_metrics(self):
        """Plot and save training and validation metrics."""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.model_name}_loss_plot.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, 'g-', label='Training Accuracy')
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, 'm-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.model_name}_accuracy_plot.png')
        plt.close()

    def _plot_confusion_matrix(self, all_preds, all_labels):
        """Plot and save confusion matrix for validation set."""
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Validation Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{self.model_name}_confusion_matrix.png')
        plt.close()

    def train(self):
        """Main training loop with fine-tuning, early stopping, and learning rate scheduling."""
        train_loader, val_loader, class_weights = self._get_data_loaders()
        model = self._create_model(fine_tune=False)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)

        best_val_accuracy = 0.0
        epochs_no_improve = 0
        best_model_path = f"{self.model_name}_best_model.pth"
        final_model_path = f"{self.model_name}_model.pth"

        for epoch in range(self.num_epochs):
            if epoch == self.fine_tune_epoch and self.fine_tune_epoch > 0:
                print("Starting fine-tuning with reduced learning rate...")
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate / 10)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)

            train_loss, train_accuracy, train_precision, train_recall, train_f1 = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_preds, val_labels = self._validate_epoch(model, val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.train_precisions.append(train_precision)
            self.val_precisions.append(val_precision)
            self.train_recalls.append(train_recall)
            self.val_recalls.append(val_recall)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)

            print(f"Epoch {epoch+1}/{self.num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            scheduler.step(val_loss)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        torch.save(model.state_dict(), final_model_path)
        print(f"Final PyTorch model saved as {final_model_path}")
        print(f"Best PyTorch model saved as {best_model_path}")

        self._plot_metrics()
        self._plot_confusion_matrix(val_preds, val_labels)

if __name__ == "__main__":
    args = BrainTumorTrainer._parse_arguments()
    trainer = BrainTumorTrainer(args)
    trainer.train()