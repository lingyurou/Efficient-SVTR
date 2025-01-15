import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SVTRCompact
from dataset import TextDataset  # Assume a custom dataset class is defined

# Hyperparameters
num_classes = 100  # Adjust based on your dataset
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Dataset and DataLoader
train_dataset = TextDataset(root="path/to/train_data")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
model = SVTRCompact(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "svtr_compact.pth")
