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
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)    #after conv becomes -> 64,16,16 -> pool -> 64,8,8
        self.conv4 = nn.Conv2d(64, 120, 3,padding=1)    #after conv becomes -> 120,8,8 ->pool -> 120,4,4
        self.fc1 = nn.Linear(120 * 4 * 4, 60)
        self.fc2 = nn.Linear(60, 10)
        # self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


#inititalize model
classifier = NeuralNetwork().to(device)

#Set Training Parameters

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)

epochs = 65
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



# Epoch 0 - Loss: 1.90754
# Epoch 1 - Loss: 1.60108
# Epoch 2 - Loss: 1.48999
# Epoch 3 - Loss: 1.41474
# Epoch 4 - Loss: 1.36318
# Epoch 5 - Loss: 1.31937
# Epoch 6 - Loss: 1.27877
# Epoch 7 - Loss: 1.23991
# Epoch 8 - Loss: 1.20844
# Epoch 9 - Loss: 1.16721
# Epoch 10 - Loss: 1.13753
# Epoch 11 - Loss: 1.10678
# Epoch 12 - Loss: 1.07483
# Epoch 13 - Loss: 1.05011
# Epoch 14 - Loss: 1.02101
# Epoch 15 - Loss: 0.99636
# Epoch 16 - Loss: 0.97127
# Epoch 17 - Loss: 0.95092
# Epoch 18 - Loss: 0.92547
# Epoch 19 - Loss: 0.91028
# Epoch 20 - Loss: 0.89260
# Epoch 21 - Loss: 0.87726
# Epoch 22 - Loss: 0.85716
# Epoch 23 - Loss: 0.84152
# Epoch 24 - Loss: 0.82679
# Epoch 25 - Loss: 0.80740
# Epoch 26 - Loss: 0.79449
# Epoch 27 - Loss: 0.77947
# Epoch 28 - Loss: 0.76511
# Epoch 29 - Loss: 0.75166
# Epoch 30 - Loss: 0.73607
# Epoch 31 - Loss: 0.72616
# Epoch 32 - Loss: 0.70955
# Epoch 33 - Loss: 0.70010
# Epoch 34 - Loss: 0.68476
# Epoch 35 - Loss: 0.67154
# Epoch 36 - Loss: 0.66046
# Epoch 37 - Loss: 0.65445
# Epoch 38 - Loss: 0.63684
# Epoch 39 - Loss: 0.62632
# Epoch 40 - Loss: 0.61949
# Epoch 41 - Loss: 0.60427
# Epoch 42 - Loss: 0.59396
# Epoch 43 - Loss: 0.58320
# Epoch 44 - Loss: 0.57351
# Epoch 45 - Loss: 0.56232
# Epoch 46 - Loss: 0.55521
# Epoch 47 - Loss: 0.54343
# Epoch 48 - Loss: 0.53488
# Epoch 49 - Loss: 0.52467
# Epoch 50 - Loss: 0.51985
# Epoch 51 - Loss: 0.50544
# Epoch 52 - Loss: 0.49802
# Epoch 53 - Loss: 0.48924
# Epoch 54 - Loss: 0.48324
# Epoch 55 - Loss: 0.47405
# Epoch 56 - Loss: 0.46771
# Epoch 57 - Loss: 0.46074
# Epoch 58 - Loss: 0.45162
# Epoch 59 - Loss: 0.44459
# Epoch 60 - Loss: 0.43405
# Epoch 61 - Loss: 0.42649
# Epoch 62 - Loss: 0.42002
# Epoch 63 - Loss: 0.41096
# Epoch 64 - Loss: 0.40698
# Validation Loss: 0.933
# Validation Accuracy: 71.830%
# approx 35min