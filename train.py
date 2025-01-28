import torch
import torch.nn as nn
import torch.optim as optim

from model import PINN
from dataloader import dataloader



model = PINN(2, 1)
loader = dataloader("") # Path to data

epochs = 150
lr = 1e-4
betas = (0.5, 0.9) # ...

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

for epoch in range(epochs):
    epoch_loss = []

    for i, (inp, tgt) in enumerate(dataloader):

        out = model(inp)

        MSEloss = criterion(tgt, out)
        PHYSloss = model.physical_loss(out)

        loss = MSEloss + PHYSloss   # TODO: Weigh physical loss more/less/...

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

