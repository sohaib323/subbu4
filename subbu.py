import streamlit as st
import pandas as pd
from sklearn.svm import SVC


st.title("Disease Detection Using SVM")


data = pd.read_csv("subbu.csv")
st.write(data)

x = data[["Temperature","Heart_Rate"]]
y = data["Disease"].map({"Yes":1,"No":0})

model = SVC()
model.fit(x,y)

temp = st.number_input("enter here temprature:",50.0,120.0,step=0.5)
het = st.number_input('enter here heart rate:',60.0,120.0,step=0.5)

prediction = model.predict([[temp,het]])[0]
res = "Yes" if prediction == 1 else "No"

st.write("Disease detected:",res)