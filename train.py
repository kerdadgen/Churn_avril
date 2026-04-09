import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import pickle

df = pd.read_csv('data/customer_churn.csv')

df_interm = df.drop(["Names", "Onboard_date", "Location", "Company", "Account_Manager"], axis=1)

# 3. Feature and Target Split
X = df_interm.drop("Churn", axis=1) # Features
y = df_interm["Churn"]            # Target variable


# 4. Data Splitting for Evaluation (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model Training (Scikit-learn Logistic Regression)
# Using class_weight='balanced' to handle potential class imbalance, similar to RandomForest
sk_logit_model = LogisticRegression(random_state=101, solver='liblinear', class_weight='balanced')
sk_logit_model.fit(X_train_scaled, y_train)
print("Scikit-learn Logistic Regression model trained.")

# 7. Model Evaluation
y_pred = sk_logit_model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 8. Save the model and scaler to pickle format
with open('model.pkl', 'wb') as f:
    pickle.dump(sk_logit_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved to pickle files.")
