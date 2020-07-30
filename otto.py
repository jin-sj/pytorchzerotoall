import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda:0"
device = torch.device(dev)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(93, 50)
        self.l2 = torch.nn.Linear(50, 25)
        self.l3 = torch.nn.Linear(25, 9)

    def forward(self, x):
        x = x.view(-1, 93)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class OttoDataset(Dataset):
    def __init__(self, file_location):
        data =  np.loadtxt(file_location, dtype=np.float32, delimiter=",", skiprows=1)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 1:-1]).to(device)
        self.y_data = torch.from_numpy(data[:, -1]).long().to(device)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

BATCH_SIZE = 64
dataset = OttoDataset("./data/otto/train.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)
model = Model()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = model(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = torch.max(output.data, 1)[1] # 0th index is value, 1st index is class
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')

def main():
    epoch = 5
    for epoch_idx in range(epoch):
        train(epoch_idx)
    test()

if __name__ == "__main__":
    main()
