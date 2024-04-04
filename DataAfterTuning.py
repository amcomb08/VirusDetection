import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the processed data
df = pd.read_csv("c:\\scaled_data.csv")

# Split data into training and test set
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
lr = LogisticRegression(max_iter=10000)

# Use the provided parameters for RandomForest
rf = RandomForestClassifier(n_estimators=150, max_features='sqrt', max_depth=20, criterion='gini', bootstrap=True)

gb = GradientBoostingClassifier(n_estimators=100)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Accuracy of Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Accuracy of Gradient Boosting:", accuracy_score(y_test, y_pred_gb))
