import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Load the processed data
df = pd.read_csv("c:\\scaled_data.csv")

# Split data into training and test set
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
#lr = LogisticRegression(max_iter=10000)
#rf = RandomForestClassifier(n_estimators=150, max_features='sqrt', max_depth=20, criterion='gini', bootstrap=True)
#gb = GradientBoostingClassifier(n_estimators=200, max_features='sqrt', max_depth=20, learning_rate= 0.1)
rf = RandomForestClassifier(n_estimators=150, max_features='sqrt', max_depth=20, criterion='gini', bootstrap=False)
gb = GradientBoostingClassifier(n_estimators=200, max_features='sqrt', max_depth=20, learning_rate= 0.1)


# Fit models
models = {'Random Forest': rf, 'Gradient Boosting': gb}
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability estimates of the positive class
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Print metrics
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.show()
