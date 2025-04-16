import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import struct
from array import array
from os.path import join
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

# Step 1: Define the MNIST data loader (only needed subset of 3 and 8)
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):        
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())        

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
            images.append(img)
        return np.array(images), np.array(labels)

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# Placeholder paths (adjust to actual file locations)
input_path = "MNISTDataset/"
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load the dataset
mnist_loader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_loader.load_data()

# Step 2: Filter only digits 3 and 8
def filter_digits(images, labels, digit1=3, digit2=8) -> Tuple[np.ndarray, np.ndarray]:
    mask = (labels == digit1) | (labels == digit2)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    binary_labels = (filtered_labels == digit2).astype(int)  # Label '8' as 1, '3' as 0
    return filtered_images, binary_labels

x_train, y_train = filter_digits(x_train, y_train)
x_test, y_test = filter_digits(x_test, y_test)

# Normalize and reshape data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Split train into train and val
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

# Convert to tensors
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# Loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# # Display few samples
# fig, axs = plt.subplots(1, 5, figsize=(15, 3))
# for i in range(5):
#     axs[i].imshow(x_train[i].reshape(28, 28), cmap="gray")
#     axs[i].set_title(f"Label: {y_train[i]}")
#     axs[i].axis("off")
# plt.tight_layout()
# plt.show()

# Define a simple MLP
class MNISTBinaryClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output logits for binary classification
        )

    def forward(self, x):
        return self.model(x)

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTBinaryClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # Validation accuracy
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu()
            all_preds.extend(preds)
            all_targets.extend(yb)
    acc = accuracy_score(all_targets, all_preds)
    print(f"Epoch {epoch+1}: Validation Accuracy = {acc*100:.2f}%")

# Test set evaluation
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_targets.extend(yb)
test_acc = accuracy_score(all_targets, all_preds)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "mnist_3vs8_classifier.pt")



# Define CFVAE from your model.py
class CFVAE(nn.Module):
    def __init__(self, feat_dim, emb_dim1, _mlp_dim1, _mlp_dim2, _mlp_dim3, mlp_inpemb, f_dim1, f_dim2):
        super(CFVAE, self).__init__()
        self.enc1 = nn.Linear(feat_dim, emb_dim1)
        self.enc2 = nn.Linear(emb_dim1, 64)
        self.dec1 = nn.Linear(32, emb_dim1)
        self.dec2 = nn.Linear(emb_dim1, feat_dim)
        self.word_embeddings = nn.Linear(feat_dim, mlp_inpemb)
        self.ln1 = nn.LayerNorm(mlp_inpemb)
        self.fc1 = nn.Linear(mlp_inpemb, f_dim1)
        self.ln2 = nn.LayerNorm(f_dim1)
        self.fc2 = nn.Linear(f_dim1, f_dim2)
        self.scorelayer = nn.Linear(f_dim2, 1)
        self.pred = nn.Linear(1, 2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = torch.relu(self.enc1(x))
        enc = self.enc2(enc).view(-1, 2, 32)
        mu, log_var = enc[:, 0, :], enc[:, 1, :]
        z = self.reparameterize(mu, log_var)
        dec = torch.relu(self.dec1(z))
        recon = self.dec2(dec)
        embed = self.word_embeddings(recon)
        embed = self.ln1(embed)
        out = torch.relu(self.fc1(embed))
        out = self.ln2(out)
        out = torch.relu(self.fc2(out))
        out = self.scorelayer(out)
        pred = self.pred(out)
        return recon, mu, log_var, pred

# Final loss function for VAE
def final_loss(bce_loss, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + KLD

# Train CFVAE
X = torch.tensor(x_train[y_train == 1], dtype=torch.float32)
Y = torch.tensor(y_train[y_train == 1], dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CFVAE(784, 64, 0, 0, 0, 64, 32, 16).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion_recon = nn.MSELoss()
criterion_cf = nn.CrossEntropyLoss()
batch_size, epochs = 32, 20

model.train()
for epoch in range(epochs):
    perm = torch.randperm(X.size(0))
    for i in range(0, X.size(0), batch_size):
        idx = perm[i:i+batch_size]
        x_batch, y_batch = X[idx].to(device), Y[idx].to(device)
        y_cf = 1 - y_batch
        optimizer.zero_grad()
        recon, mu, logvar, pred = model(x_batch)
        recon_loss = criterion_recon(recon, x_batch)
        kld = final_loss(recon_loss, mu, logvar)
        cf_loss = criterion_cf(pred, y_cf)
        loss = kld + 50 * cf_loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Visualize
model.eval()
with torch.no_grad():
    sample = X[:5].to(device)
    recon, _, _, _ = model(sample)

fig, axs = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axs[0, i].imshow(sample[i].cpu().numpy().reshape(28, 28), cmap="gray")
    axs[0, i].set_title("Original 8")
    axs[0, i].axis("off")
    axs[1, i].imshow(recon[i].cpu().numpy().reshape(28, 28), cmap="gray")
    axs[1, i].set_title("Counterfactual → 3")
    axs[1, i].axis("off")
plt.tight_layout()
plt.show()
