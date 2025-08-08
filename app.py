import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("onlinefoods.csv")

# Preprocessing
if 'Unnamed: 12' in df.columns:
    df = df.drop(columns=['Unnamed: 12'])

le = LabelEncoder()
df['Output'] = le.fit_transform(df['Output'])

X = df.drop('Output', axis=1)
y = df['Output']

categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

numeric_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Latih model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Prediksi Output Online Food Order")

# Buat input form dinamis berdasarkan kolom
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Input {col}", step=1.0)

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    prediction = model.predict(input_df)
    label = le.inverse_transform(prediction)[0]
    st.success(f"Prediksi Output: {label}")

