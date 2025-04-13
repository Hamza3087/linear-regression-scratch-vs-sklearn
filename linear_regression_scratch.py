import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set random seed for reproducibility
np.random.seed(42)

class LinearRegressionScratch:
    """Linear Regression implementation from scratch using Ordinary Least Squares (OLS)"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Compute weights using the Normal Equation"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
    
    def predict(self, X):
        """Predict values"""
        return X.dot(self.weights) + self.bias
    
    def mean_squared_error(self, y_true, y_pred):
        """Compute the Mean Squared Error (MSE)"""
        return np.mean((y_true - y_pred) ** 2)

# Generate synthetic dataset
def generate_synthetic_data(n_samples=1000, noise_min=0.1, noise_max=100.0):
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([0.5, 0.7])
    true_bias = 2.0
    noise_levels = np.random.uniform(noise_min, noise_max, n_samples)
    noise = np.array([nl * np.random.randn() for nl in noise_levels])
    y = X.dot(true_weights) + true_bias + noise
    return X, y, true_weights, true_bias, noise_levels

# Output directory
output_dir = "linear_regression_results"
os.makedirs(output_dir, exist_ok=True)

# Generate data
X, y, true_weights, true_bias, noise_levels = generate_synthetic_data(n_samples=1000, noise_min=0.1, noise_max=1.0)

# Save synthetic data
data_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
data_df['target'] = y
data_df['noise_level'] = noise_levels
data_df.to_csv(os.path.join(output_dir, "synthetic_data.csv"), index=False)

# Save true parameters
params_df = pd.DataFrame({'parameter': ['weight_1', 'weight_2', 'bias'], 'value': [true_weights[0], true_weights[1], true_bias]})
params_df.to_csv(os.path.join(output_dir, "true_parameters.csv"), index=False)

# Normalize dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
norm_data_df = pd.DataFrame(X_scaled, columns=['feature_1_normalized', 'feature_2_normalized'])
norm_data_df['target'] = y
norm_data_df['noise_level'] = noise_levels
norm_data_df.to_csv(os.path.join(output_dir, "normalized_data.csv"), index=False)

# Train-test split
X_train, X_test, y_train, y_test, noise_train, noise_test = train_test_split(X_scaled, y, noise_levels, test_size=0.2, random_state=42)

# Save train and test sets
train_df = pd.DataFrame(X_train, columns=['feature_1_normalized', 'feature_2_normalized'])
train_df['target'] = y_train
train_df['noise_level'] = noise_train
train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)

test_df = pd.DataFrame(X_test, columns=['feature_1_normalized', 'feature_2_normalized'])
test_df['target'] = y_test
test_df['noise_level'] = noise_test
test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

# Train Linear Regression from scratch
lr_scratch = LinearRegressionScratch()
lr_scratch.fit(X_train, y_train)
y_pred_scratch = lr_scratch.predict(X_test)
mse_scratch = lr_scratch.mean_squared_error(y_test, y_pred_scratch)
r2_scratch = r2_score(y_test, y_pred_scratch)

# Train Scikit-Learn Linear Regression
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted_scratch': y_pred_scratch,
    'predicted_sklearn': y_pred_sklearn,
    'residual_scratch': y_test - y_pred_scratch,
    'residual_sklearn': y_test - y_pred_sklearn,
    'noise_level': noise_test
})
predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

# Save model parameters and metrics
results_df = pd.DataFrame({
    'model': ['Scratch', 'Scikit-Learn', 'True Values'],
    'weight_1': [lr_scratch.weights[0], lr_sklearn.coef_[0], true_weights[0]],
    'weight_2': [lr_scratch.weights[1], lr_sklearn.coef_[1], true_weights[1]],
    'bias': [lr_scratch.bias, lr_sklearn.intercept_, true_bias],
    'mse': [mse_scratch, mse_sklearn, None],
    'r2_score': [r2_scratch, r2_sklearn, None]
})
results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_scratch, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Scratch Model: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_sklearn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Scikit-Learn Model: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
residuals_scratch = y_test - y_pred_scratch
plt.scatter(y_pred_scratch, residuals_scratch, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Scratch Model: Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
residuals_sklearn = y_test - y_pred_sklearn
plt.scatter(y_pred_sklearn, residuals_sklearn, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Scikit-Learn Model: Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plots.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(noise_levels, bins=20, alpha=0.7)
plt.title('Distribution of Noise Levels')
plt.xlabel('Noise Level')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "noise_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(noise_test, np.abs(residuals_scratch), alpha=0.5, label='Scratch Model')
plt.scatter(noise_test, np.abs(residuals_sklearn), alpha=0.5, label='Scikit-Learn Model')
plt.title('Absolute Error vs. Noise Level')
plt.xlabel('Noise Level')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "error_vs_noise.png"), dpi=300)
plt.show()

print("\nAll files have been saved to:", output_dir)
