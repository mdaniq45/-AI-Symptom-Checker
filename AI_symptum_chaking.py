import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Load Dataset
df = pd.read_csv(r"Disease precaution.csv")  # Replace with actual dataset path

# Fill missing values
df.fillna("No Symptom", inplace=True)

# Combine all symptom columns into one
df["Combined_Symptoms"] = df.iloc[:, 1:].apply(lambda x: ' '.join(x.values), axis=1)
df = df[['Disease', 'Combined_Symptoms']]

# Convert text data into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Combined_Symptoms"])

# Encode target variable
y = LabelEncoder().fit_transform(df["Disease"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naïve Bayes": MultinomialNB(),
    "SVM": SVC()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Streamlit Web App
st.title("🩺 AI Symptom Checker")
user_input = st.text_input("Enter symptoms (comma-separated):")

if st.button("Predict Disease"):
    symptoms_vector = vectorizer.transform([user_input])
    predictions = {name: model.predict(symptoms_vector)[0] for name, model in models.items()}
    
    st.write("### Predictions:")
    for name, pred in predictions.items():
        st.write(f"**{name}:** {LabelEncoder().fit(df['Disease']).inverse_transform([pred])[0]}")

# Run the app using: streamlit run filename.py
