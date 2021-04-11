# HR-Analytics-Predict-who-will-change-jobs
Predict whether an applicant will search for a new job or stay with the company, as well as interpret the impact of various factors on an employee's decision.

- Developed a webapp using streamlit and heroku which predicts whether an applicant is looking for a new data science job or not with an Accuracy of 84.0%.
- Preprocessed the data to a specific data form which is suitable for machine learning models to learn.
- Performed Exploratory Data Analysis to visualize, summarize and interpret the hidden information in the dataset.
- Applied Dummy Encoding to all Categorical Features.
- Performed several machine learning algorithms such as Support Vector Machine, Decision Tree, Random Forest, Logistic Regression, KNN, and XG Boost.
- Improved the performance of Logistics Regression & Random Forest models by re-training them with SMOTE, a technique to synthesis new samples in minority class (in our case its class 1).

### Data Cleaning

- Removed null values from training dataset.
- Changed the input values to specific required data type of values in 'enrolled_university', 'experience', and 'company_size' columns.

### Exploratory Data Analysis

- Compared train and test dataset with eachother with ordered categorical data as it gives more control over visualization.
 
