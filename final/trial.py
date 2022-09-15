import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.model_selection import train_test_split

#from sklearn.decompostion.PCA import PCA
from sklearn.metrics import accuracy_score


from PIL import Image

st.title('EMPLOYEE ATTRITION PREDICTION')
st.subheader('USING DATASCIENCE')

image=Image.open('employee1.jpg')
st.image(image,use_column_width=True)


dataset_name=st.sidebar.selectbox('select dataset',('EmployeeAttrition',''))
#dataset_name=st.sidebar.selectbox('Select dataset',('Breast Cancer','Iris','Wine'))
classifier_name=st.sidebar.selectbox('select classifier',('','Randomforest','decision tree','naive bayes','logistic regresssion'))





upload=st.file_uploader("choose a csv",type='csv')
if upload is not None:
    data=pd.read_csv(upload)
    st.write(data)
    st.success("Sucessfully Loaded Dataset")
    st.write("shape of the datset: ")
    st.write(data.shape)

    if st.button("check null values"):
        st.write(data.isnull().sum())


    if st.button("sum of null values: "):
	    st.write(data.isnull().sum().sum())
	    st.write(data.isnull().values.any())


    st.write("describing the datset")
    st.write(data.describe())

    st.info("Attrition of the company: ")
    x=data['Attrition'].value_counts()
    st.write(x)




    a=st.sidebar.selectbox('Visualization',('','Countplot','Barplot'))
    
    if a=='Countplot':
        sns.countplot(data['Attrition'])
        st.write("------------------------")
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    elif a=='Barplot':
	    x.plot.pie(autopct="%.1f%%")
	    st.pyplot()
	    st.set_option('deprecation.showPyplotGlobalUse', False)




    if st.button("Datatypes and Values"):
		    for column in data.columns:
			    if data[column].dtype == object:
				    st.write(str(column) + ' : ' + str(data[column].unique()))
				    st.write(data[column].value_counts())
				    st.write("-----------------------------------------------------------")

	


    data = data.drop('EmployeeNumber', axis = 1)
    data = data.drop('Over18', axis = 1)
    data = data.drop('OverTime', axis = 1)
    data = data.drop('EmployeeCount', axis = 1)

    showPyplotGlobalUse = False



    if st.button("Correlation of the dataset"):
        st.write(data.corr())


showPyplotGlobalUse = False


   



	


    

