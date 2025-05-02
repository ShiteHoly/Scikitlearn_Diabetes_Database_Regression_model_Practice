import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,train_test_split
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# 1. Load and scale data
diabetes = load_diabetes()
x = diabetes.data  # shape (442, 10)
y = diabetes.target  # shape (442,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_trainval, X_test, y_trainval, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# get a final test set, which we will use after Cross-Validation train

# 2. Hyperparameters
epochs = 500
lr = 0.01
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 3. Model Definition
class DiabetesModel(nn.Module):
    def __init__(self, hidden_size=64):
        super(DiabetesModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

# 4. Cross-validation loop
lr_list = [1e-3,1e-2,1e-1]
hidden_layer_list = [32, 64, 128]

best_config = None
best_cv_mse = float('inf')

for lr in lr_list:
    for layer in hidden_layer_list:
        mse_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_trainval), 1):
            # a) Split data for this fold
            x_train = torch.tensor(x_trainval[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y_trainval[train_idx], dtype=torch.float32).view(-1, 1)
            x_test  = torch.tensor(x_trainval[val_idx],  dtype=torch.float32)
            y_test  = torch.tensor(y_trainval[val_idx],  dtype=torch.float32).view(-1, 1)

            model     = DiabetesModel(layer)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # c) Training
            for epoch in range(epochs):
                model.train()
                preds = model(x_train)
                loss  = criterion(preds, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # d) Evaluate on this fold's test set
            model.eval()
            with torch.no_grad():
                preds = model(x_test)
                mse   = criterion(preds, y_test).item()
            mse_scores.append(mse)
            print(f"Fold {fold}/{n_splits} - Val MSE: {mse:.4f}")

        # 5. Summarize cross-validation results
        mean_mse = np.mean(mse_scores)
        std_mse  = np.std(mse_scores)
        print(f"lr={lr}, hidden layer={layer}, CV MSE: {mean_mse:.4f} ± {std_mse:.4f}\n")
        if mean_mse < best_cv_mse:
            best_cv_mse = mean_mse
            best_config = (lr, layer)

print(f"Best config: lr={best_config[0]}, hidden={best_config[1]}(CV MSE={best_cv_mse:.4f})\n")

# 6. Retrain final model on full trainval set
lr, layer = best_config
x_full = torch.tensor(x_trainval, dtype=torch.float32)
y_full = torch.tensor(y_trainval, dtype=torch.float32).view(-1, 1)
criterion = nn.MSELoss()
model = DiabetesModel()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    predictions = model(x_full)
    loss = criterion(predictions, y_full)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 7. Final evaluation on the test set
model.eval()
with torch.no_grad():
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
    test_predictions = model(X_test)
    mse = criterion(test_predictions, Y_test)
print(f"Test MSE: {mse:.4f}")

# Save the final trained model's state dictionary
model_save_path = 'best.pt'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# … after printing CV MSE …
import matplotlib.pyplot as plt
# 1. Prepare x-axis as fold numbers 1…n_splits
folds = np.arange(1, len(mse_scores) + 1)

# 2. Plot the per-fold MSE
plt.figure()
plt.plot(folds, mse_scores, marker='o', linestyle='-')
# 3. Draw a horizontal line at the mean
plt.hlines(mean_mse, 1, n_splits,
           colors='red', linestyles='--',
           label=f'Mean MSE = {mean_mse:.2f}')
# 4. Label and style
plt.xlabel('Fold')
plt.ylabel('Test MSE')
plt.title('5-Fold Cross-Validation MSE')
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.show()

# Parity plot for final test
y_true = Y_test.numpy().flatten()
y_pred = test_predictions.numpy().flatten()
plt.figure()
plt.scatter(y_true, y_pred, alpha=0.6)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, 'k--', label='Perfect')
plt.xlabel('Actual progression')
plt.ylabel('Predicted progression')
plt.title('Parity Plot on Test Set')
plt.legend()
plt.grid(True)
plt.show()