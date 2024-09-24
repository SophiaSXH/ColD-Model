import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, \
                            roc_auc_score, matthews_corrcoef, confusion_matrix, \
                            precision_recall_curve, cohen_kappa_score


# Step 1: Define your dataset class and apply transformations
class CollagenDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_idx = idx // 100
        sub_idx = idx % 100
        image = Image.open(self.image_paths[img_idx])
        label = self.labels[img_idx]

        sub_width = image.width // 10
        sub_height = image.height // 10

        i = sub_idx % 10
        j = sub_idx // 10

        left = i * sub_width
        upper = j * sub_height
        right = left + sub_width
        lower = upper + sub_height

        image = image.crop((left, upper, right, lower))

        if self.transform:
            image = self.transform(image)
        return image, label

# Step 2: Load image paths and labels
image_folder = '/path_to_AFM_images/AFM image'
weights_path = '/path_to_downloaded_pre-train_model_weight/mobilenet_v2-b0353104.pth'
num_epochs = 25


image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
image_paths = [os.path.join(image_folder, f) for f in image_files]

image_paths = np.repeat(image_paths, 100)
# labels = np.concatenate([np.repeat(np.ones(50), 100), np.repeat(np.zeros(50), 100)])
# Create labels based on image names
image_numbers = [int(os.path.splitext(name)[0]) for name in image_files]
labels = [1 if 1 <= number <= 50 else 0 for number in image_numbers]
labels = np.repeat(labels, 100)

# Step 3: Split your dataset
train_val_images, test_images, train_val_labels, test_labels = train_test_split(image_paths,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_val_images,
                                                                      train_val_labels,
                                                                      test_size=1 / 8,
                                                                      random_state=42)

# Step 4: Data resize and to tensor
transform_train = transforms.Compose([transforms.Resize((51, 51)),
                                      transforms.ToTensor()])

transform_test = transforms.Compose([transforms.Resize((51, 51)),
                                     transforms.ToTensor()])

# Step 5: Create dataset objects and data loaders
train_dataset = CollagenDataset(train_images, train_labels, transform=transform_train)
val_dataset = CollagenDataset(val_images, val_labels, transform=transform_test)
test_dataset = CollagenDataset(test_images, test_labels, transform=transform_test)

batch_size = 4
num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Step 6: Load pre-trained model
mps_device = torch.device("mps")
model = torchvision.models.mobilenet_v2(weights=None)
model.load_state_dict(torch.load(weights_path))

# Change the last layer to output 2 classes
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
model.to(mps_device)

# Step 7: Train the VGG model and obtain training accuracy
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_accuracies, test_accuracies = [], []
tpr_list, fpr_list, precision_list, recall_list = [], [], [], []
precisions, recalls, f1_scores, aucs = [], [], [], []
evaluation_scores = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(torch.float32).to(mps_device), labels.to(torch.float32).to(mps_device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_acc = train_correct / total
    train_accuracies.append(train_acc)

    # Evaluation
    model.eval()
    test_correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(torch.float32).to(mps_device), labels.to(torch.float32).to(mps_device)

            outputs = model(images)
            loss = criterion(outputs, labels.long())

            _, predicted = outputs.max(1)
            total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_acc = test_correct / total
    test_accuracies.append(test_acc)

# Step 8: compute evaluation metrics
    precision_value, recall_value, _ = precision_recall_curve(all_labels, all_predictions)
    precision_score_value = precision_score(all_labels, all_predictions)
    recall_score_value = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)

    tpr_list.append(tpr)
    fpr_list.append(fpr)
    precisions.append(precision_score_value)
    recalls.append(recall_score_value)
    f1_scores.append(f1)
    aucs.append(auc)

    # Append the evaluation scores to the list
    evaluation_scores.append({
        'Epoch': epoch + 1,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Precision': precision_score_value,
        'Recall': recall_score_value,
        'FPR': fpr,
        'TPR': tpr,
        'F1 Score': f1,
        'AUC': auc,

    })

    # Convert the list of dictionaries to a DataFrame
    evaluation_df = pd.DataFrame(evaluation_scores)

    # Save the DataFrame to an Excel file
    excel_file_path = os.path.join(desktop_path, "evaluation_scores_mobilenet_v2.xlsx")
    evaluation_df.to_excel(excel_file_path, index=False)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Precision: {precision_score_value:.4f}, Recall: {recall_score_value:.4f}, "
          f"F1 Score: {f1:.4f}, AUC: {auc:.4f}")

# Step 9: Plot the results
epochs = range(1, num_epochs + 1)

# Plot training and testing accuracy
plt.figure()
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, test_accuracies, label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.show()

# Plot Precision-Recall curve, F1 score, and ROC AUC score
plt.figure()
plt.plot(epochs, precisions, label="Precision")
plt.plot(epochs, recalls, label="Recall")
plt.plot(epochs, f1_scores, label="F1 Score")
plt.plot(epochs, aucs, label="ROC AUC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend(loc='lower right')
plt.ylim(0.5, 1)
plt.yticks(np.arange(0.5, 1.1, 0.1))
plt.show()

# Plot the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Collagen", "Non-Collagen"], rotation=45)
plt.yticks(tick_marks, ["Collagen", "Non-Collagen"])

thresh = cm_normalized.max() / 2.0
for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
    plt.text(j, i, "{:.2f}".format(cm_normalized[i, j]),
             horizontalalignment="center",
             color="white" if cm_normalized[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
plt.figure()
for epoch, (tpr, fpr) in enumerate(zip(tpr_list, fpr_list), start=1):
    plt.plot(fpr, tpr, label=f"Epoch {epoch}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# mathew's correlation coefficient
mcc = matthews_corrcoef(all_labels, all_predictions)
print(f"MCC: {mcc:.4f}")

# cohens kappa
kappa = cohen_kappa_score(all_labels, all_predictions)
print(f"Cohen's Kappa: {kappa:.4f}")

# Step 10: Save the model
torch.save(model.state_dict(), os.path.join(desktop_path, "ColD_model.pt"))
