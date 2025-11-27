import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Cache dataset loading
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df["species"])

st.title("🌸 Iris Flower Classification App")
st.write("Adjust features from the sidebar to predict the flower species.")

# Sidebar input
st.sidebar.title("Input Features")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df["sepal length (cm)"].min()),
    float(df["sepal length (cm)"].max()),
    float(df["sepal length (cm)"].mean()),
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(df["sepal width (cm)"].min()),
    float(df["sepal width (cm)"].max()),
    float(df["sepal width (cm)"].mean()),
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(df["petal length (cm)"].min()),
    float(df["petal length (cm)"].max()),
    float(df["petal length (cm)"].mean()),
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(df["petal width (cm)"].min()),
    float(df["petal width (cm)"].max()),
    float(df["petal width (cm)"].mean()),
)

# Make prediction using DataFrame (fixes warnings)
input_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=df.columns[:-1])

prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display result
st.subheader("🌼 Predicted Species")
st.write(f"### **{predicted_species}**")

st.write("---")
st.write("Made with ❤️ using Streamlit and scikit-learn.")
