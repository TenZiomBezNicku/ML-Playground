import torch
import torch.nn.functional as F
from torch import nn

x = torch.rand(1000, 2) * 2 - 1
y = ((x[:, 0] ** 2 + x[:, 1] ** 2) < 0.25).float()
y = y.unsqueeze(1)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(3000):
    optimizer.zero_grad()

    loss = F.binary_cross_entropy_with_logits(model(x), y)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}; loss: {loss.item()}")

x_test = torch.rand(1000, 2) * 2 - 1
y_test = ((x_test[:, 0] ** 2 + x_test[:, 1] ** 2) < 0.25).float()
y_test = y_test.unsqueeze(1)

with torch.no_grad():
    predictions = (torch.sigmoid(model(x_test)) >= 0.5).float()
    accuracy = (predictions == y_test).float().mean()

print(f"Accuracy: {accuracy.item() * 100:.2f}%")