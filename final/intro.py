import streamlit as st

st.title("EMPLOYEE ATTRITION PREDICTION")

from PIL import Image

st.subheader("using DATASCIENCE")

image=Image.open("employee1.jpg")
st.image(image,use_column_width=True)

st.write("hello")
st.markdown("this is a markdown cell")
st.success("congrats man")
st.warning("carefull dude")
st.info("this is info")
st.error("this is error")
st.help(range)

import numpy as np
import pandas as pd

dataframe=np.random.rand(10,20)
st.dataframe(dataframe)
st.text("---"*100)
st.line_chart(dataframe)
st.area_chart(dataframe)
st.bar_chart(dataframe)

if st.button("say hello"):
    st.write("hello nitish how are u")
else:
	st.write("sorry gudbye")

x=st.radio("what is your name?",{'nitish','lochan','harsha','tarun'})

if x=='nitish':
	st.write("hi niti")
elif x=='lochan':
	st.write("hi lochan")
elif x=='harsha':
	st.write("hi harsha")
else :
	st.write("hi tarun")

age=st.slider("how old are you?",0,20,45)

st.write("your age is",age)

n=st.number_input("enter number")
st.write("number is ",n)


upload=st.file_uploader("choose a csv",type='csv')

if upload is not None:
	data=pd.read_csv(upload)
	st.write(data)
	st.success("sucess man")
else:
	st.warning("u hve not uploaded")

color=st.sidebar.color_picker("pick color",'#FF0000')
st.write("color is ",color)


side=st.sidebar.selectbox("select your algo",('knn','naive','random forest','decision tree'))
#st.write()

import time

my_bar=st.progress(0)
for percent in range(100):
	time.sleep(0.1)
	my_bar.progress(percent+1)

with st.spinner('wait for it ..'):
     time.sleep(5)
st.success('succesfull')

st.balloons()