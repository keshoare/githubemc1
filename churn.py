#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Task 1: Data Preparation )
import pandas as pd
import numpy as np


file_path = r"C:\MLProjects\sklearn.xlsx"
df = pd.read_excel(file_path)


print("Dataset Shape (Rows, Columns):", df.shape)
print("\nMissing Values per Column:\n", df.isnull().sum())
print("\nData Types before preprocessing:\n", df.dtypes)


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())  # fixed here


df = df.drop("customerID", axis=1)


cat_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


print("\nAfter Preprocessing:")
print("Dataset Shape:", df.shape)
print("\nData Types after preprocessing:\n", df.dtypes)


# In[ ]:


output_path = r"C:\MLProjects\prepared_data.xlsx"
df.to_excel(output_path, index=False)

print(f"Preprocessed dataset saved at: {output_path}")


# In[10]:


#Task 2
import pandas as pd
from sklearn.model_selection import train_test_split


file_path = r"C:\MLProjects\prepared_data.xlsx"
df = pd.read_excel(file_path)


X = df.drop("Churn_Yes", axis=1)   
y = df["Churn_Yes"]           


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Training Set Shape:", X_train.shape, y_train.shape)
print("Testing Set Shape:", X_test.shape, y_test.shape)


# In[11]:


# task 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


file_path = r"C:\MLProjects\prepared_data.xlsx"
df = pd.read_excel(file_path)


X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


feature_importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)


print("Top 10 Important Features:")
print(feature_importances.head(10))


# In[12]:


#Task 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


file_path = r"C:\MLProjects\prepared_data.xlsx"
df = pd.read_excel(file_path)


X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")


# In[13]:


# Task 5: Model Training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


file_path = r"C:\MLProjects\prepared_data.xlsx"
df = pd.read_excel(file_path)


X = df.drop("Churn_Yes", axis=1)  
y = df["Churn_Yes"]               


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model training completed successfully!")


# In[14]:


# Task 6: Model Evaluation

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


y_pred = model.predict(X_test_scaled) 
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)


print("Model Evaluation Results:")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"ROC-AUC Score : {roc_auc:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))





