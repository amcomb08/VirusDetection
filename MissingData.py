import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier


# Load your data
df = pd.read_csv('c:\\train_data.csv')

# Drop rows with missing data
#df = df.dropna()
df.fillna(0, inplace=True)

# Drop Flow ID column since it is a concatenation of other columns
df.drop('Flow ID', axis=1, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)

# scale columns
cols_to_scale = [' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
       ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Fwd Header Length.1','Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min']




#Initialize scaler
scaler = StandardScaler()

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
#Encode class column to 0 and 1
df['Class'] = df['Class'].map({'Benign': 0, 'Trojan': 1})

# Frequency encoding for Source IP
source_ip_freq = df[' Source IP'].value_counts().to_dict()
df[' Source IP'] = df[' Source IP'].map(source_ip_freq)

# Frequency encoding for Destination IP
dest_ip_freq = df[' Destination IP'].value_counts().to_dict()
df[' Destination IP'] = df[' Destination IP'].map(dest_ip_freq)

df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], format='%d/%m/%Y %H:%M:%S')

df['IsWeekend'] = df[' Timestamp'].dt.dayofweek.isin([4, 5, 6]).astype(int)  # 1 for Friday, Saturday & Sunday (weekend), 0 for weekday

df = df.drop(columns=[' Timestamp'])

X = df.drop('Class', axis=1)
y = df['Class']

# Using RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=200, max_features='log2', max_depth=None, criterion='entropy', bootstrap=False,min_samples_split=10,min_samples_leaf=2)
clf_rf = clf_rf.fit(X, y)

# Using ExtraTreesClassifier
clf_et = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', max_depth=None, criterion='gini', bootstrap=True,min_samples_split=5,min_samples_leaf=1)
clf_et = clf_et.fit(X, y)

# Using GradientBoostingClassifier
clf_gb = GradientBoostingClassifier(n_estimators=200, max_features='log2', max_depth=15, learning_rate= 0.1,subsample= 0.9,min_samples_split=5,min_samples_leaf=2)
clf_gb = clf_gb.fit(X, y)

# Extract feature importances from classifiers
feature_importances_rf = clf_rf.feature_importances_
feature_importances_et = clf_et.feature_importances_
feature_importances_gb = clf_gb.feature_importances_

# Create a DataFrame to hold the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest': feature_importances_rf,
    'Extra Trees': feature_importances_et,
    'Gradient Boosting': feature_importances_gb
})

# Calculate average importance across classifiers for sorting
feature_importance_df['Average Importance'] = feature_importance_df[['Random Forest', 'Extra Trees', 'Gradient Boosting']].mean(axis=1)
feature_importance_df = feature_importance_df.sort_values(by='Average Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(15, 10))
sns.barplot(data=feature_importance_df, x='Average Importance', y='Feature', palette='viridis')
plt.title('Combined Feature Importances from Different Classifiers')
plt.tight_layout()
plt.show()

# Retrieve important features from each model
important_features_rf = set(X.columns[(SelectFromModel(clf_rf, prefit=True).get_support())])
important_features_et = set(X.columns[(SelectFromModel(clf_et, prefit=True).get_support())])
important_features_gb = set(X.columns[(SelectFromModel(clf_gb, prefit=True).get_support())])

# Combine all the important features into one set
all_important_features = important_features_rf.union(important_features_gb)

all_important_features.add('Class')

# Drop columns not in the set of important features
columns_to_drop = [col for col in df.columns if col not in all_important_features]
df.drop(columns=columns_to_drop, inplace=True)

df.to_csv("c:\\scaled_data.csv", index=False)