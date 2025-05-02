import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),#Hidden layers to learn complex hidden features:
                                                   #expand the data into 64 features to give the model more capacity
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
        )
    def forward(self, x):
        return self.net(x)

model = DiabetesModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    model.train()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)

    model.eval()
    with torch.no_grad():
        predictions_val = model(x_val)
        val_loss = criterion(predictions_val, y_val)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1} / {epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(x_test)
    loss = criterion(predictions, y_test)
    print(f'Test Loss MSE: {loss.item():.4f}')

import matplotlib.pyplot as plt

plt.scatter(y_test.numpy(), predictions.numpy())
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.show()