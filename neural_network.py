import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#GETTING AND LOADING DATA

device = torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#split train set into train and validation set
trainSet_size = int(0.8 * len(training_set))
valSet_size = len(training_set) - trainSet_size
trainSet, valSet = torch.utils.data.random_split(training_set, [trainSet_size, valSet_size])

#load train and validation set
train_loader = DataLoader(trainSet, batch_size=50, shuffle=True)
val_loader = DataLoader(valSet, batch_size=50, shuffle=False)

#load test set
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=50, shuffle=False)

#Neural Network Architecture

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)    #after conv becomes -> 16,32,32 ->pool -> 16,16,16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)   #after conv becomes -> 32,16,16
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)    #after conv becomes -> 64,16,16 ->pool -> 64,8,8
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#inititalize model
classifier = NeuralNetwork().to(device)

#Set Training Parameters

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

epochs = 75
losses=[]
# try:
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = classifier(images)
        loss = lossFn(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch} - Loss: {epoch_loss:.5f}")

torch.save(classifier.state_dict(), "model_interrupted.pth")

#Evaluation

classifier.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)
        loss = lossFn(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)

print(f"Validation Loss: {avg_val_loss:.3f}")
accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.3f}%")

#plot the loss v epoch graph

plt.plot(range(epochs), losses)
plt.plot(48, [avg_val_loss], "g+")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()