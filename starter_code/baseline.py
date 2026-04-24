import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import joblib
import os

# 1. Load Data
train_df = pd.read_csv('data/train.csv')
test_df  = pd.read_csv('data/test.csv')

# 2. Create Target Variable (Pass=1 if result>=10, Fail=0)
# Already present as 'result' column in train.csv

# 3. Preprocessing - Encode categorical columns
test_ids = test_df['id'].copy()
test_df  = test_df.drop(columns=['id'])

cat_cols = [col for col in train_df.select_dtypes(include='object').columns]
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col]  = le.transform(test_df[col].astype(str))

# 4. Define Features & Target
X = train_df.drop('result', axis=1)
y = train_df['result']

# 5. Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Baseline Model - Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluate on Validation Set
predictions = model.predict(X_val)
print(f"Baseline F1-Score:  {f1_score(y_val, predictions):.4f}")
print(classification_report(y_val, predictions))

# 8. Save Model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/baseline_student_model.pkl')
print("✅ Model saved to models/baseline_student_model.pkl")

# 9. Generate Submission File
final_predictions = model.predict(test_df)

os.makedirs('submission', exist_ok=True)
submission = pd.DataFrame({
    'id':     test_ids,
    'result': final_predictions
})
submission.to_csv('submission/submission.csv', index=False)
print(f"✅ Submission saved to submission/submission.csv ({len(submission)} rows)")
print(submission.head())
