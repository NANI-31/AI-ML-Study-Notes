import streamlit as st

st.title("This is my first app :)")
st.write("Hello welcome to streamlit")

name = st.text_input("what is your name?")

if name:
    st.success(f"Welcome {name}")

age = st.slider("select your age", 10,100, 18)

st.write("you have selected the age : ", age)

location = st.selectbox("select your location", ["karachi", "islamabad", "lahore"])

st.write("you have selected the location : ", location)

submit = st.button("Submit")

if submit:
    st.write("Your information has been submitted successfully")

if st.button("Click me"):
    st.balloons()