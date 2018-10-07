import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, models, transforms

import sys; sys.path.append('./functions')

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 7)
    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg13(pretrained=True)
vgg.classifier = vgg.classifier[0]
vgg = vgg.to(device)
for param in vgg.parameters():
    param.requires_grad = False

# hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001


model = Classifier().to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        ])

train_dataset = datasets.ImageFolder(root='./functions/data/classifier/train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
test_dataset = datasets.ImageFolder(root='./functions/data/classifier/test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size, shuffle=True,
                                          num_workers=0)

def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        features = vgg(images)
        outputs = model(features)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)

    return train_loss

def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            features = vgg(images)
            outputs = model(features)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total

    return val_loss, val_acc

loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))

    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

# save the trained model
np.save('loss_list.npy', np.array(loss_list))
np.save('val_loss_list.npy', np.array(val_loss_list))
np.save('val_acc_list.npy', np.array(val_acc_list))
torch.save(model.state_dict(), './functions/data/classifier.pth')
