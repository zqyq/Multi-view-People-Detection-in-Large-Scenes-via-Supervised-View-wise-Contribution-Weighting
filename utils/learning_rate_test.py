import matplotlib.pyplot as plt
import torch
import torch.optim
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 1)

    def forward(self, x):
        return self.net(x)


model = Net()
lr = 1e-4
lr_decay = 2e-4
x = torch.rand(3, 10)
# epoch=200
gt = torch.ones(3, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + lr_decay * epoch) ** epoch)

lr_y = []
lr_x = []
for epoch in range(200):
    lr_x.append(epoch)
    lr_y.append(optimizer.state_dict()['param_groups'][0]['lr'])
    optimizer.zero_grad()
    res = model(x)
    loss = F.mse_loss(res, gt)
    loss.backward()
    optimizer.step()
    scheduler.step()

plt.plot(lr_x, lr_y)
plt.show()
