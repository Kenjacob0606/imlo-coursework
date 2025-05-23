import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#GETTING AND LOADING DATA
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.48, 0.45, 0.4), (0.229, 0.224, 0.225))])
device = torch.device("cpu")
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
        self.fc1 = nn.Linear(120 * 8 * 8, 60)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#load trained model
new_classifier = NeuralNetwork().to(device)
new_classifier.load_state_dict(torch.load("classifier.pth"))

#Evaluation with test set
lossFn = nn.CrossEntropyLoss()
test_loss = 0
correct = 0
total = 0
new_classifier.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = new_classifier(images)
        loss = lossFn(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    avg_test_loss = test_loss / len(test_loader)

print(f"Final Validation Loss: {avg_test_loss:.3f}")
accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.3f}%")