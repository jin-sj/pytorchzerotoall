import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


#x_data = Variable(torch.Tensor([[2.1, 0.1], [4.2, 0.8], [3.1, 0.9], [3.3, 0.2]]))
#y_data = Variable(torch.Tensor([[0.], [1.], [0.], [1.]]))

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("data/diabetes.csv", delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, -1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        y_pred = torch.sigmoid(self.linear3(x))
        #x = torch.nn.functional.relu(self.linear1(x))
        #x = torch.nn.functional.relu(self.linear2(x))
        #y_pred = torch.nn.functional.relu(self.linear3(x))
        return y_pred

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
model = Model()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        y_preds = model(inputs)

        loss = criterion(y_preds, labels)
        print(epoch, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

hour_var = Variable(torch.Tensor([[0.2881, 0.33115, 0.1283, -0.12391232, -0.87420, 0.4701, 0.192, -0.65921]]))
print("predict 1 hour", 1.0, model(hour_var).data[0][0] > 0.5)
#print("predict 1 hour", 1.0, model(hour_var).data[0][0] > 0.5)
#hour_var = Variable(torch.Tensor([[3.2, 0.5]]))
#print("predict 7 hour", 7.0, model(hour_var).data[0][0] > 0.5)





