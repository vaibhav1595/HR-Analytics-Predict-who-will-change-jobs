import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from time import sleep
from stqdm import stqdm
np.random.seed(42)

st.set_page_config(layout="wide")

st.write("""
# HR Analytics App
This app predicts whether an applicant will search for a new job or stay with the company!
Data obtained from the [Kaggle](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists).
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe

Gender = st.sidebar.selectbox('Gender',('Male', 'Female', 'Other'))
City_development_index = st.sidebar.slider('City_development_index', min_value=0.00, max_value= 1.00)
Relevant_Experience = st.sidebar.selectbox('Relevant_Experience', ('Has relevent experience', 'No relevent experience'))
Total_Experience = st.sidebar.slider('Total_Experience', min_value=0, max_value= 21)
Enrolled_university = st.sidebar.selectbox('Enrolled_university', ('No Enrollment', 'Full time course', 'Part time course'))
Education_level = st.sidebar.selectbox('Education_level', ('Graduate', 'Masters', 'High School', 'Phd', 'Primary School'))
Company_size = st.sidebar.selectbox('Company_size', ('0-9', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'))
Company_type = st.sidebar.selectbox('Company_type', ('Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'NGO', 'Other'))
Major_discipline = st.sidebar.selectbox('Major_discipline', ('STEM', 'Humanities', 'Business Degree', 'Arts', 'Other', 'No Major'))


# Create dataframe obtained from input features similar to dataset
data = {'Gender': Gender,
        'City_development_index': City_development_index,
        'Relevant_Experience': Relevant_Experience,
        'Enrolled_university': Enrolled_university,
        'Education_level': Education_level,
        'Company_size': Company_size,
        'Company_type': Company_type,
        'Major_discipline': Major_discipline,
        'Total_Experience': Total_Experience}

features = pd.DataFrame(data, index=[0])

# Show input features
columns = ['Gender','City_development_index','Relevant_Experience','Enrolled_university','Education_level',
          'Company_size','Company_type','Major_discipline']

st.write('### **User Input Values**')
# for i in range(len(columns)):
#     st.write(str(list(data.keys())[i]))
#     st.write(data[str(columns[i])])
st.write(features)

# Load the HR Analytics dataset
hr_data = pd.read_csv('hr_Data.csv')
hr_data.dropna(inplace=True)
X = hr_data.iloc[:, :-1]
y = hr_data.iloc[:,-1:]


# Create a final dataframe with input data and existing dataset
final = pd.concat([features, X])
Final_copy = final.copy()
Final_copy = pd.get_dummies(final, columns=columns)


if st.button('Predict'):

    # Building ML Model
    sm = SMOTE()
    X_sm, y_sm = sm.fit_resample(Final_copy[1:], y)
    sm_rf = RandomForestClassifier(n_estimators=500)
    sm_rf.fit(X_sm, y_sm)

    prediction = sm_rf.predict(Final_copy[:1])

    Output = ''
    if prediction == 1:
        Output = 'Yes, the person is looking for a job.'
    else:
        Output = 'No, the person is not looking for a job.'

    st.write(Output)
    # success message
    st.success('This is a success message!')
    st.empty()

# RemoveWARNING: :
st.set_option('deprecation.showPyplotGlobalUse', False)
