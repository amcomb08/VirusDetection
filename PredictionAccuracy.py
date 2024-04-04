import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from dateutil import parser
from sklearn.model_selection import train_test_split


# Load training data
df_train = pd.read_csv('C:\\train_data.csv')

# Split df_train into train and test subsets
df_train, df_test = train_test_split(df_train, test_size=0.5, random_state=42)

# Combining both for frequency encoding
all_data = pd.concat([df_test, df_train], axis=0)
source_ip_freq = all_data[' Source IP'].value_counts().to_dict()
dest_ip_freq = all_data[' Destination IP'].value_counts().to_dict()


# Drop columns if they exist in either df_test or df_train
for column in ['Flow ID', 'Unnamed: 0']:
    if column in df_test.columns:
        df_test.drop(column, axis=1, inplace=True)
    if column in df_train.columns:
        df_train.drop(column, axis=1, inplace=True)


# Frequency encoding for Source IP and Destination IP
for df in [df_test, df_train]:
    df[' Source IP'] = df[' Source IP'].map(source_ip_freq).fillna(0)
    df[' Destination IP'] = df[' Destination IP'].map(dest_ip_freq).fillna(0)
    df[' Timestamp'] = df[' Timestamp'].apply(lambda x: parser.parse(x))
    df['IsWeekend'] = df[' Timestamp'].dt.dayofweek.isin([4, 5, 6]).astype(int)
    df.drop(columns=[' Timestamp'], inplace=True)


# Fill the NaN values in both df_test and df_train with 0
df_test.fillna(0, inplace=True)
df_train.fillna(0, inplace=True)


# List of columns to scale
cols_to_scale = [
    ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
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
       ' Idle Max', ' Idle Min'
]


# Initialize scaler
scaler = StandardScaler()


# Applying the scaling
df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])


# Define target variable encoding
df_train['Class'] = df_train['Class'].map({'Benign': 0, 'Trojan': 1})


X_train = df_train.drop('Class', axis=1)
y_train = df_train['Class']


# Training classifiers
clf_rf = RandomForestClassifier(n_estimators=200, max_features='log2', max_depth=None, criterion='entropy', bootstrap=False, min_samples_split=10, min_samples_leaf=2)
clf_rf.fit(X_train, y_train)


clf_et = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', max_depth=None, criterion='gini', bootstrap=True, min_samples_split=5, min_samples_leaf=1)
clf_et.fit(X_train, y_train)


clf_gb = GradientBoostingClassifier(n_estimators=200, max_features='log2', max_depth=15, learning_rate=0.1, subsample=0.9, min_samples_split=5, min_samples_leaf=2)
clf_gb.fit(X_train, y_train)


# Retrieve important features from each model
important_features_rf = set(X_train.columns[(SelectFromModel(clf_rf, prefit=True).get_support())])
important_features_et = set(X_train.columns[(SelectFromModel(clf_et, prefit=True).get_support())])
important_features_gb = set(X_train.columns[(SelectFromModel(clf_gb, prefit=True).get_support())])


# Combine all the important features
all_important_features = important_features_rf.union(important_features_gb).union(important_features_et)


# Drop columns not in the set of important features for both train and test data
columns_to_drop_train = [col for col in df_train.columns if col not in all_important_features.union({'Class'})]
columns_to_drop_test = [col for col in df_test.columns if col not in all_important_features]

# Drop columns not in the set of important features for the train data
df_train.drop(columns=columns_to_drop_train, inplace=True)

# Update X_train and y_train
X_train = df_train.drop('Class', axis=1)
y_train = df_train['Class']

# Refit the model on modified X_train
clf_gb.fit(X_train, y_train)

# Drop columns not in the set of important features for the test data
df_test.drop(columns=columns_to_drop_test, inplace=True)

# Predict using the model on df_test
predictions = clf_gb.predict(df_test)

# Convert numerical predictions to string format
predictions_mapped = ['Benign' if pred == 0 else 'Trojan' for pred in predictions]

# Add predictions to df_test
df_test['Class'] = predictions_mapped

# Define target variable encoding for df_test
df_test['Class'] = df_test['Class'].map({'Benign': 0, 'Trojan': 1})

# Compare predictions to actual values for accuracy
accuracy = (predictions == df_test['Class']).mean()

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the df_test with predictions
df_test.to_csv("C:\\df_with_predictions_for_accuracy.csv", index=False)


