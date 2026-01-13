import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"
]

df = pd.read_csv('Data/KDDTest+.txt', names=columns)

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Categorical Encoding
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = le.fit_transform(df[col])

x_test = df.drop(['label', 'difficulty'], axis=1)
y_test = df['label']

# Standardizing
scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(x_test)

x_test_reshaped = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

model = load_model("model.keras")

# Generate Predictions
y_pred_probs = model.predict(x_test_reshaped)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Header
print("\n" + "="*50)
print(" "*20 + "REPORT")
print("="*50)

# Basic Metrics
print(f"OVERALL ACCURACY: {accuracy_score(y_test, y_pred):.4%}")

# Print Classification Report (Precision, Recall, F1)
print("\nDETAILED STATS:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Confusion matrix
print("CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)

# Terminal display
cm_df = pd.DataFrame(cm, index=['Actual Normal', 'Actual Anomaly'], columns=['Pred Normal', 'Pred Anomaly'])

print(cm_df)
print("="*50)