import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

# Data Processing and Augmentation
class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.data = SVHN(root=f'E:/cisc3024/{split}', split=split, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, label

def get_transforms(train=True, max_rotation=15, scale=(0.8, 1.0)):
    if train:
        return A.Compose([
            A.RandomResizedCrop(32, 32, scale=scale),
            A.Rotate(limit=max_rotation),
            A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
            ToTensorV2()
        ])

# Neural Network Setup (VGG Model)
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # Update input channels to 3
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),  # Adjust the input size based on the final feature map size
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    model.train()
    train_losses = []
    test_losses = []
    f1_scores = []
    roc_auc_macro = []
    roc_auc_micro = []

    model_path = "E:/cisc3024/checkpoints/best.pth"
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        model.eval()
        running_test_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                probs = nn.functional.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate average test loss
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # Calculate F1 score, ROC AUC (Macro and Micro)
        f1 = f1_score (all_labels, all_preds, average='macro')
        f1_scores.append(f1)

        # Use 'ovr' (one-vs-rest) strategy for multi-class AUC calculation
        roc_auc_macro_val = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        roc_auc_macro.append(roc_auc_macro_val)

        roc_auc_micro_val = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='micro')
        roc_auc_micro.append(roc_auc_micro_val)

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC (Macro): {roc_auc_macro_val:.4f}, ROC AUC (Micro): {roc_auc_micro_val:.4f}")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), model_path)
            print("最佳模型已保存到:", model_path)

    return train_losses, test_losses, f1_scores, roc_auc_macro, roc_auc_micro


def evaluate_model(model, test_loader, criterion, num_epochs=10):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Get the predicted labels
            _, preds = torch.max(outputs, 1)

            # Append to lists
            all_preds.extend(preds.cpu().numpy())  # Convert to numpy
            all_labels.extend(labels.cpu().numpy())  # Convert to numpy
            all_probs.extend(probs.cpu().numpy())  # Convert to numpy

        # Convert lists to numpy arrays after the loop
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Calculate ROC AUC for each class
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        # Calculate average loss
        avg_loss = running_loss / len(test_loader.dataset)

        # Calculate accuracy
        accuracy = np.mean(all_preds == all_labels)

        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f'Accuracy: {accuracy}, ROC AUC: {roc_auc}')

    return all_labels, all_preds, all_probs, accuracy


# Main execution
train_dataset = SVHNDataset(split='train', transform=get_transforms(train=True))
test_dataset = SVHNDataset(split='test', transform=get_transforms(train=False))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = VGG(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-4)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.85, weight_decay=1e-4)
# Additional Analysis and Visualization
def plot_training_curves(train_losses, test_losses):
    num_epochs = 20
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))

    # Annotate the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(all_labels, all_probs, num_classes=10):
    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_f1_scores(f1_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, label='F1 Score', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_roc_auc(roc_auc_macro, roc_auc_micro):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(roc_auc_macro) + 1), roc_auc_macro, label='ROC AUC (Macro)', color='blue')
    plt.plot(range(1, len(roc_auc_micro) + 1), roc_auc_micro, label='ROC AUC (Micro)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_class_performance(all_labels, all_preds, num_classes=10):
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    plt.figure()
    plt.bar(range(num_classes), class_accuracy)
    plt.xticks(range(num_classes), [str(i) for i in range(num_classes)])
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 1)
    plt.show()

# Main execution with added metrics
train_losses, test_losses, f1_scores, roc_auc_macro, roc_auc_micro = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)
# Evaluating the model
all_labels, all_preds, all_probs, accuracy = evaluate_model(model, test_loader, criterion)

# Plotting result
plot_training_curves(train_losses, test_losses)
plot_confusion_matrix(all_labels, all_preds)
plot_roc_curve(all_labels, all_probs)
plot_class_performance (all_labels, all_preds, num_classes=10)
plot_f1_scores(f1_scores)
plot_roc_auc(roc_auc_macro, roc_auc_micro)
