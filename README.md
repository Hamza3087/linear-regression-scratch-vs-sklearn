# linear-regression-scratch-vs-sklearn

This project demonstrates the implementation of Linear Regression from scratch using Ordinary Least Squares (OLS) and compares its performance with Scikit-Learn’s built-in LinearRegression model.

The goal is to understand the core mechanics of linear regression by building it manually, and then comparing its performance to a well-optimized library implementation.

📌 Objectives
Build a Linear Regression model from scratch using Python.

Compare it with Scikit-Learn’s LinearRegression.

Evaluate and visualize performance metrics.

🗂️ Task Breakdown
1. Dataset Preparation
Generate a synthetic dataset with:

Two independent variables: x1, x2

One dependent variable: y

Add random noise for realism.

Normalize the dataset and split it into:

Training set: 80%

Test set: 20%

2. Implement Linear Regression (OLS) from Scratch
Create a class LinearRegressionScratch with the following methods:

fit(X, y):
Compute weights using the Normal Equation.

predict(X):
Predict values for given inputs.

mean_squared_error(y_true, y_pred):
Compute the Mean Squared Error (MSE).

3. Compare with Scikit-Learn
Train and evaluate both your custom model and Scikit-Learn’s LinearRegression on the same dataset.

Compute and compare:

Mean Squared Error (MSE)

R² score

Learned parameters (weights & bias)

4. Visualization
Plot Actual vs Predicted values for both models.

Create a residual plot to analyze the error distribution.

🛠️ Tools & Libraries
Python 3.x

NumPy

Pandas

Matplotlib / Seaborn

Scikit-Learn

🚀 How to Run
Clone the repository:
git clone https://github.com/Hamza3087/linear-regression-scratch-vs-sklearn.git
cd linear-regression-scratch-vs-sklearn

Install dependencies:
pip install -r requirements.txt

python linear_regression_scratch.py


📁 Files to include in your repo:
linear-regression-scratch-vs-sklearn/
│
├── README.md
├── requirements.txt
├── linear_regression_scratch_vs_sklearn.ipynb


📊 Results
Compare both models visually and numerically.

Understand how well your manual implementation performs against Scikit-Learn.

🎯 Learning Outcomes
Deep understanding of Linear Regression fundamentals.

Hands-on experience with model evaluation and visualization.

Insights into model performance comparison and interpretation.

📎 License
This project is open-source and free to use under the MIT License.

