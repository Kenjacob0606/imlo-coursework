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
train_loader = DataLoader(trainSet, batch_size=64, shuffle=True)
val_loader = DataLoader(valSet, batch_size=64, shuffle=False)

#load test set
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

#Neural Network Architecture

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)    #after conv becomes -> 16,32,32
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)   #after conv becomes -> 32,32,32 ->pool -> 32,16,16
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)    #after conv becomes -> 64,16,16
        self.conv4 = nn.Conv2d(64, 120, 3,padding=1)    #after conv becomes -> 100,16,16 ->pool -> 100,8,8
        self.fc1 = nn.Linear(120 * 8 * 8, 10)
        # self.fc2 = nn.Linear(120, 10)
        # self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x


#inititalize model
classifier = NeuralNetwork().to(device)

#Set Training Parameters

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)

epochs = 48
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

# classifier.eval()
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

# plot the loss v epoch graph

plt.plot(range(epochs), losses)
plt.plot(epochs, [avg_val_loss], "g+")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()



# Epoch 0 - Loss: 1.85374
# Epoch 1 - Loss: 1.51614
# Epoch 2 - Loss: 1.41797
# Epoch 3 - Loss: 1.35645
# Epoch 4 - Loss: 1.31103
# Epoch 5 - Loss: 1.26362
# Epoch 6 - Loss: 1.21913
# Epoch 7 - Loss: 1.17825
# Epoch 8 - Loss: 1.13432
# Epoch 9 - Loss: 1.09679
# Epoch 10 - Loss: 1.05514
# Epoch 11 - Loss: 1.02525
# Epoch 12 - Loss: 0.99229
# Epoch 13 - Loss: 0.96318
# Epoch 14 - Loss: 0.93845
# Epoch 15 - Loss: 0.91480
# Epoch 16 - Loss: 0.89538
# Epoch 17 - Loss: 0.87370
# Epoch 18 - Loss: 0.85213
# Epoch 19 - Loss: 0.83650
# Epoch 20 - Loss: 0.81969
# Epoch 21 - Loss: 0.79909
# Epoch 22 - Loss: 0.78152
# Epoch 23 - Loss: 0.76587
# Epoch 24 - Loss: 0.74833
# Epoch 25 - Loss: 0.73289
# Epoch 26 - Loss: 0.71825
# Epoch 27 - Loss: 0.70378
# Epoch 28 - Loss: 0.68873
# Epoch 29 - Loss: 0.67081
# Epoch 30 - Loss: 0.65732
# Epoch 31 - Loss: 0.64315
# Epoch 32 - Loss: 0.62826
# Epoch 33 - Loss: 0.61115
# Epoch 34 - Loss: 0.60118
# Epoch 35 - Loss: 0.58354
# Epoch 36 - Loss: 0.57367
# Epoch 37 - Loss: 0.55655
# Epoch 38 - Loss: 0.54475
# Epoch 39 - Loss: 0.52916
# Epoch 40 - Loss: 0.52055
# Epoch 41 - Loss: 0.50681
# Epoch 42 - Loss: 0.48956
# Epoch 43 - Loss: 0.48056
# Epoch 44 - Loss: 0.46679
# Epoch 45 - Loss: 0.45497
# Epoch 46 - Loss: 0.44101
# Epoch 47 - Loss: 0.42838
# Validation Loss: 0.890
# Validation Accuracy: 72.160%