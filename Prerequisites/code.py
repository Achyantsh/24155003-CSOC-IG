# Imports
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
import json

#  Load and Parse Data
with open("data/housing-complete.csv", 'r') as f:
    lines = f.readlines()

column_names = lines[0].strip().split(',')
target = 'median_house_value'
features = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income'
]

feature_indexes = [column_names.index(c) for c in features]
target_index = column_names.index(target)
# Convert to float and clean by skipping incomplete rows
data = []
for line in lines[1:]:
    values=line.strip().split(',')

    missing=False
    for i in feature_indexes+[target_index]:
        if(i>=len(values) or values[i].strip()==''):
            missing=True
            break
    if missing:
        continue

    feature_values=[]
    for i in feature_indexes:
        feature_values.append(float(values[i].strip()))
    data.append(feature_values)

# Split into training (20k) and test (rest) ---
train_data = data[:20000]
test_data = data[20000:]

X_train = [row[:-1] for row in train_data]
y_train = [row[-1] for row in train_data]

X_test_raw = [row[:-1] for row in test_data]
y_test = [row[-1] for row in test_data]

# Normalize (min-max using TRAIN stats only) ---
def get_min_max(X):
    num_features = len(X[0])
    mins = [min([row[i] for row in X]) for i in range(num_features)]
    maxs = [max([row[i] for row in X]) for i in range(num_features)]
    return mins, maxs

def normalize_given(X, mins, maxs):
    normalized = []
    for row in X:
        new_row = []
        for i in range(len(row)):
            if maxs[i] == mins[i]:
                new_row.append(0.0)
            else:
                val = (row[i] - mins[i]) / (maxs[i] - mins[i])
                new_row.append(val)
        normalized.append(new_row)
    return normalized

mins, maxs = get_min_max(X_train)
X_train = normalize_given(X_train, mins, maxs)
X_test = normalize_given(X_test_raw, mins, maxs)
# Code for Implementation type 1: Pure Python
# ----------- Dot Product -----------
def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

# ----------- Compute Cost -----------
def compute_cost(X, y, Ws, b):
    n = len(X)
    total_error = 0.0
    for i in range(n):
        y_pred = dot_product(Ws, X[i]) + b
        total_error += (y_pred - y[i]) ** 2
    return total_error / (2 * n)

# ----------- Gradient Descent -----------
def gradient_descent(X, y, alpha=0.01, epsilon=0.001, max_itr=1000):
    m = len(X[0])       # number of features
    n = len(X)          # number of samples
    Ws = [0.0] * m      # weights
    b = 0.0             # bias

    previous_cost = float('inf')
    cost_history = []

    for itr in range(max_itr):
        dw = [0.0] * m
        db = 0.0

        # Compute gradients
        for i in range(n):
            y_pred = dot_product(Ws, X[i]) + b
            error = y_pred - y[i]
            for j in range(m):
                dw[j] += error * X[i][j]
            db += error

        # Update weights and bias
        for j in range(m):
            Ws[j] -= (alpha * dw[j]) / n
        b -= (alpha * db) / n

        # Compute and store cost
        current_cost = compute_cost(X, y, Ws, b)
        cost_history.append(current_cost)

        if abs(previous_cost - current_cost) < epsilon:
            print(f"Converged at iteration {itr}, Cost = {current_cost:.6f}")
            break

        previous_cost = current_cost

        if itr % 100 == 0:
            print(f"Iteration {itr}: Cost = {current_cost:.6f}")

    return Ws, b, cost_history

# ----------- Predict Function -----------
def predict(x, Ws, b):
    return dot_product(x, Ws) + b

# Start the timer
start_time_pure = time.time()

# Train the model
Ws_pure, b_pure, cost_history_pure = gradient_descent(X_train, y_train, alpha=0.001, epsilon=0.0001, max_itr = 2000)

# End the timer
end_time_pure = time.time()

# Compute elapsed time
elapsed_time_pure = end_time_pure - start_time_pure
def evaluate_model(y_true, y_pred):
    n = len(y_true)
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    rmse = math.sqrt(mse)
    mean_y = sum(y_true) / n
    ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot)
    return mae, rmse, r2


y_pred_pure = [predict(x, Ws_pure, b_pure) for x in X_test]

# Evaluate on test set
mae_pure, rmse_pure, r2_pure = evaluate_model(y_test, y_pred_pure)
print(f"[Pure Python] Test MAE: {mae_pure:.2f}")
print(f"[Pure Python] Test RMSE: {rmse_pure:.2f}")
print(f"[Pure Python] Test R2 Score: {r2_pure:.4f}")
print(f"[Pure Python] Training Time: {elapsed_time_pure:.4f} seconds")
# Plot cost vs. iterations
plt.figure(figsize=(8, 5))
plt.plot(cost_history_pure, label='Cost over Iterations (Pure Python)', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence of Cost Function (Pure Python)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Code for Implementation type 2: Using Numpy
def compute_cost_np(X, y, W, b):
    n = X.shape[0]
    y_pred = X @ W + b
    error = y_pred - y
    cost = (1 / (2 * n)) * np.sum(error ** 2)
    return cost

def gradient_descent_np(X, y, alpha=0.01, epsilon=0.001, max_itr=1000):
    n, m = X.shape
    W = np.zeros(m)
    b = 0.0
    cost_history = []
    prev_cost = float('inf')

    for itr in range(max_itr):
        y_pred = X @ W + b
        error = y_pred - y
        dW = (1 / n) * (X.T @ error)
        db = (1 / n) * np.sum(error)

        W -= alpha * dW
        b -= alpha * db

        current_cost = compute_cost_np(X, y, W, b)
        cost_history.append(current_cost)

        if abs(prev_cost - current_cost) < epsilon:
            print(f"Converged at iteration {itr}, Cost = {current_cost:.6f}")
            break

        prev_cost = current_cost
        if itr % 100 == 0:
            print(f"Iteration {itr}: Cost = {current_cost:.6f}")

    return W, b, cost_history

# Convert your normalized lists to NumPy arrays
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

# Start the timer
start_time_np = time.time()

# Train
W_np, b_np, cost_history_np = gradient_descent_np(X_train_np, y_train_np, alpha=0.001, epsilon=0.0001, max_itr = 2000)

# End the timer
end_time_np = time.time()

elapsed_time_np = end_time_np - start_time_np

# Predict
y_pred_np = X_test_np @ W_np + b_np

def evaluate_np(y_true, y_pred):
    n = len(y_true)
    mae = np.mean(abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mae, rmse, r2

mae_np, rmse_np, r2_np = evaluate_np(y_test_np, y_pred_np)

print(f"[Numpy] Test MAE: {mae_np:.2f}")
print(f"[Numpy] Test RMSE: {rmse_np:.2f}")
print(f"[Numpy] Test R2 Score: {r2_np:.4f}")
print(f"[Numpy] Training Time: {elapsed_time_np:.4f} seconds")

plt.figure(figsize=(8, 5))
plt.plot(cost_history_np, label='Cost over Iterations (NumPy)', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence of Cost Function (NumPy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Code for Implementation type 3: Using Scikit-learn
# Train the model
model = LinearRegression()
start_time = time.time()
model.fit(X_train_np, y_train_np)
end_time = time.time()
sklearn_training_time = end_time - start_time
# Predict and evaluate
y_pred_sklearn = model.predict(X_test_np)
mse_sklearn = mean_squared_error(y_test_np, y_pred_sklearn)
rmse_sklearn = math.sqrt(mse_sklearn)
r2_sklearn = r2_score(y_test_np, y_pred_sklearn)
mae_sklearn = mean_absolute_error(y_test_np, y_pred_sklearn)

print(f"[Scikit-learn] Training Time: {sklearn_training_time:.4f} seconds")
print(f"[Scikit-learn] MAE: {mae_sklearn:.2f}")
print(f"[Scikit-learn] RMSE: {rmse_sklearn:.2f}")
print(f"[Scikit-learn] R² Score: {r2_sklearn:.4f}")

# Comparative Graphs and results
results = {
    'Pure Python': {
        'MAE': mae_pure,
        'RMSE': rmse_pure,
        'R2': r2_pure,
        'Time': elapsed_time_pure,
    },
    'NumPy': {
        'MAE': mae_np,
        'RMSE': rmse_np,
        'R2': r2_np,
        'Time': elapsed_time_np,
    },
    'Scikit-learn': {
        'MAE': mae_sklearn,
        'RMSE': rmse_sklearn,
        'R2': r2_sklearn,
        'Time': sklearn_training_time,
    }
}

# Save as result.json
with open('result.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to result.json")

methods = list(results.keys())
mae_values = [results[m]['MAE'] for m in methods]
rmse_values = [results[m]['RMSE'] for m in methods]
r2_values = [results[m]['R2'] for m in methods]
time_values = [results[m]['Time'] for m in methods]

x = np.arange(len(methods))
width = 0.25

# Create side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# ----- Left Plot: MAE, RMSE, R² -----
axs[0].bar(x - width, mae_values, width, label='MAE', color='skyblue')
axs[0].bar(x, rmse_values, width, label='RMSE', color='lightgreen')
axs[0].bar(x + width, r2_values, width, label='R² Score', color='salmon')
axs[0].set_xticks(x)
axs[0].set_xticklabels(methods)
axs[0].set_ylabel("Metric Value")
axs[0].set_title("Regression Metrics Comparison")
axs[0].legend()
axs[0].grid(True, axis='y', linestyle='--', alpha=0.6)

# ----- Right Plot: Time Elapsed -----
axs[1].bar(methods, time_values, color='orchid')
axs[1].set_ylabel("Time (seconds)")
axs[1].set_title("Training Time Comparison")
axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
