import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the processed data
df = pd.read_csv("c:\\scaled_data.csv")

# Split data into training and test set
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the training set into training and validation sets
X_train_base, X_val, y_train_base, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Initialize base models
rf = RandomForestClassifier(n_estimators=150, max_features='sqrt', max_depth=20, criterion='gini', bootstrap=False)
gb = GradientBoostingClassifier(n_estimators=200, max_features='sqrt', max_depth=20, learning_rate=0.1)

# Train base models on the training base dataset
rf.fit(X_train_base, y_train_base)
gb.fit(X_train_base, y_train_base)

# Predict on the validation set
val_pred_rf = rf.predict(X_val)
val_pred_gb = gb.predict(X_val)

# Combine the predictions to train the meta-learner
stacked_predictions = np.column_stack((val_pred_rf, val_pred_gb))
lr = LogisticRegression(max_iter=10000)
lr.fit(stacked_predictions, y_val)

# Predict on the test set using base models
test_pred_rf = rf.predict(X_test)
test_pred_gb = gb.predict(X_test)

# Combine the base models' test predictions to feed into the meta-learner
stacked_test_predictions = np.column_stack((test_pred_rf, test_pred_gb))
final_predictions = lr.predict(stacked_test_predictions)

# Print metrics
print("Final Stacking Classifier Results:")
print("Accuracy:", accuracy_score(y_test, final_predictions))
print("F1 Score:", f1_score(y_test, final_predictions))
print("Precision:", precision_score(y_test, final_predictions))
print("Recall:", recall_score(y_test, final_predictions))

# Plot ROC curve
y_proba = lr.predict_proba(stacked_test_predictions)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Stacking Classifier')
plt.legend(loc="lower right")
plt.show()
