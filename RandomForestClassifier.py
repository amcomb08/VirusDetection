import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("c:\\scaled_data.csv")

X = df.drop('Class', axis=1)  # All columns except 'Class'
y = df['Class']

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Take a random subset of the training data for hyperparameter tuning
subset_size = 0.5
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=subset_size, random_state=42)

# Initialize the Random Forest Classifier
clf_rf = RandomForestClassifier()

# Define the parameters and the values you want to test
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV with the model, the parameter distribution, number of iterations, and the cross-validation strategy (e.g., 5 folds)
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=1, random_state=42)

# Fit the model using the subset
random_search.fit(X_train_subset, y_train_subset)

# Print the best parameters
print(f"Best Parameters: {random_search.best_params_}")

# Predict using the best model on the original test set
y_pred = random_search.best_estimator_.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Hyperparameter Tuning: {accuracy}")
