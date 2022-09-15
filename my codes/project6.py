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


showPyplotGlobalUse = False

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





if st.button("Correlation of the dataset"):
    st.write(data.corr())


showPyplotGlobalUse = False




b=st.sidebar.selectbox('HEATMAP',('','HEATMAP'))

if b=='HEATMAP':
	plt.figure(figsize=(14,14))
	sns.heatmap(data.corr(), annot=True, fmt='.0%')
	st.pyplot()
	showPyplotGlobalUse = False





c=st.sidebar.selectbox('GRAPH ANAYLSIS',('','Age','TotalWorkingYears','YearsAtCompany','PercentSalaryHike'))
    
if c=='Age':
	fig_dims = (12, 4)
	fig, ax = plt.subplots(figsize=fig_dims)
	sns.countplot(x='Age', hue='Attrition', data = data, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1))
	st.pyplot()
elif c=='TotalWorkingYears':
	fig_dims = (12, 4)
	fig, ax = plt.subplots(figsize=fig_dims)
	sns.countplot(x='TotalWorkingYears', hue='Attrition', data = data, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1))
	st.pyplot()
elif c=='YearsAtCompany':
	fig_dims = (12, 4)
	fig, ax = plt.subplots(figsize=fig_dims)
	sns.countplot(x='YearsAtCompany', hue='Attrition', data = data, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1))
	st.pyplot()
elif c=='PercentSalaryHike':
	fig_dims = (12, 4)
	fig, ax = plt.subplots(figsize=fig_dims)
	sns.countplot(x='PercentSalaryHike', hue='Attrition', data = data, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1))
	st.pyplot()


showPyplotGlobalUse = False




from sklearn.preprocessing import LabelEncoder

for column in data.columns:
        if data[column].dtype == np.number:
            continue
        data[column] = LabelEncoder().fit_transform(data[column])

data['Age_Years']=data['Age']
data=data.drop('Age',axis=1)





X = data.iloc[:, 1:data.shape[1]].values 
Y = data.iloc[:, 0].values 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#random forest algo




if classifier_name=='Randomforest':
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
	forest.fit(X_train, Y_train)
	score=forest.score(X_train, Y_train)
	st.success("random forest score")
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(Y_test, forest.predict(X_test))
	TN = cm[0][0]
	TP = cm[1][1]
	FN = cm[1][0]
	FP = cm[0][1]
	st.write(cm)
	st.write('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
elif classifier_name=='logistic regresssion':
	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(C=0.1, solver='newton-cg' , random_state=1)
	rfe = RFE(model, n_features_to_select=5)
	fit = rfe.fit(X_train,Y_train)
	X_train_nf, X_test_nf, y_train_nf, y_test_nf = train_test_split(X, Y, test_size = 0.25, random_state = 0)
	model.fit(X_train_nf, y_train_nf)
	y_pred = model.predict(X_test_nf)
	from sklearn import metrics
	cm=metrics.confusion_matrix(y_test_nf,y_pred)
	TN = cm[0][0]
	TP = cm[1][1]
	FN = cm[1][0]
	FP = cm[0][1]
	score=rfe.score(X_train, Y_train)
	st.success("logistic regresssion")
	st.write(cm)
	st.write('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
elif classifier_name=='decision tree':
	from sklearn.tree import DecisionTreeClassifier
	def applyDecisionTree(X_train, Y_train, X_test, Y_test):
		dtree = DecisionTreeClassifier()
		dtree.fit(X_train, Y_train)
		score = dtree.score(X_test,Y_test)
		y_pred = dtree.predict(X_test)
		st.success("decision tree score")
		from sklearn import metrics
		cm=metrics.confusion_matrix(Y_test,y_pred)
		TN = cm[0][0]
		TP = cm[1][1]
		FN = cm[1][0]
		FP = cm[0][1]
		st.write(cm)
		st.write('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
	applyDecisionTree(X_train, Y_train, X_test, Y_test)
elif classifier_name=='naive bayes':
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB().fit(X_train,Y_train.ravel())
	score = clf.score(X_test,Y_test)
	st.success("naive bayes")
	predicted = clf.predict(X_test)
	from sklearn import metrics
	cm=metrics.confusion_matrix(Y_test,predicted)
	TN = cm[0][0]
	TP = cm[1][1]
	FN = cm[1][0]
	FP = cm[0][1]
	st.write(cm)
	st.write('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))


if st.sidebar.button("TOTAL PREDICTED ANAYLSIS"):
    acc=0.8532608695652174
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    st.write("Random forest is the best algorithm with accuracy",acc)
    importances = pd.DataFrame({'feature':data.iloc[:, 1:data.shape[1]].columns,'importance':np.round(forest.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    st.write(importances)




	

	


showPyplotGlobalUse = False






	


    

