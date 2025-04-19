import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter grid for GridSearchCV (wider range for nu, including more kernels)
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Different kernels
    'nu': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8],         # Valid nu values in (0, 1]
    'gamma': ['scale', 'auto']                      # Gamma values for kernel choice
}

# Instantiate the NuSVC model with higher max_iter
model = NuSVC(max_iter=5000)  # Increased max_iter

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters found through grid search
print("Best Parameters:", grid_search.best_params_)

# Train with the best parameters found
best_model = grid_search.best_estimator_

# Generate learning curve data for the best model
train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_train, y_train,
    cv=10,  # 10-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPUs for parallel computation
    train_sizes=np.linspace(0.1, 1.0, 10)  # 10 different training set sizes
)

# Compute the mean and std of training and validation scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve with shaded regions for standard deviations
plt.figure(figsize=(10, 6))

# Plot the mean training and validation scores
plt.plot(train_sizes, train_mean, label='Training Score', color='blue', lw=2)
plt.plot(train_sizes, test_mean, label='Cross-Validation Score', color='green', lw=2)

# Add shaded regions to show the variability (standard deviation)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')

# Add titles and labels
plt.title('Learning Curve for NuSVC Model with Hyperparameter Tuning', fontsize=14)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc="best")
plt.grid(True)

# Show the plot
plt.show()
