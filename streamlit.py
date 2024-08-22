import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder

# target_encoder = TargetEncoder()

# Load the model
with open('RF_model', 'rb') as file:
    model = pickle.load(file)
with open('target_encoder.pkl', 'rb') as encoder_file:
    target_encoder = pickle.load(encoder_file)

st.title('Bank Marketing Term Deposit Prediction')

# User inputs
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
education = st.selectbox('Education', ['basic 4y', 'basic 6y', 'basic 9y', 'high school', 'illiterate', 'professional course', 'university degree', 'unknown'])
default = st.selectbox('Default', ['no', 'yes', 'unknown'])
housing = st.selectbox('Housing Loan', ['no', 'yes', 'unknown'])
loan = st.selectbox('Personal Loan', ['no', 'yes', 'unknown'])
month = st.selectbox('Month of Last Contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Day of the Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
duration = st.number_input('Duration of Last Contact (in seconds)', min_value=0, value=200)
campaign = st.number_input('Number of Contacts During This Campaign', min_value=1, value=2)
pdays = st.number_input('Number of Days Since Last Contact from Previous Campaign', min_value=-1, value=10)
previous = st.number_input('Number of Contacts Performed Before This Campaign', min_value=0, value=1)
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
poutcome = st.selectbox('Outcome of the Previous Marketing Campaign', ['failure', 'nonexistent', 'success'])

# Convert user inputs to a DataFrame
input_data = {
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'contact': [contact],
    'poutcome': [poutcome]
}
input_df = pd.DataFrame(input_data)

# Map categorical variables
for col in ['housing', 'default', 'loan']:
    input_df[col] = input_df[col].map({'yes': 1, 'no': 0, 'unknown': -1})

# Ordinal encode 'education' and 'marital'
categories_education = [
    'unknown',
    'illiterate',
    'basic 4y',
    'basic 6y',
    'basic 9y',
    'high school',
    'professional course',
    'university degree'
]
categories_marital = [
    'unknown',
    'divorced',
    'married',
    'single'
]
ordinal_encoder_education = OrdinalEncoder(categories=[categories_education])
ordinal_encoder_marital = OrdinalEncoder(categories=[categories_marital])

input_df['education'] = ordinal_encoder_education.fit_transform(input_df[['education']])
input_df['marital'] = ordinal_encoder_marital.fit_transform(input_df[['marital']])

# One-hot encode 'contact' and 'poutcome'
input_df = pd.get_dummies(input_df, columns=['contact', 'poutcome'], drop_first=True)

# Map 'month' and 'day_of_week' to numerical values
month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_dict = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
input_df['month'] = input_df['month'].map(month_dict)
input_df['day_of_week'] = input_df['day_of_week'].map(day_dict)


# target = pd.Series([1], index=input_df.index)

input_df['job'] = target_encoder.transform(input_df[['job']])

# Ensure all expected columns are present
expected_columns = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'month',
                    'day_of_week', 'duration', 'campaign', 'pdays', 'previous',
                    'contact_telephone', 'poutcome_nonexistent', 'poutcome_success']

for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Drop the 'default' column, as it wasn't part of the model training
input_df = input_df.drop(columns=['default'])

# Ensure input DataFrame columns are in the same order as during training
input_df = input_df[expected_columns]

# Make predictions
prediction = model.predict(input_df)

# Display the prediction
if prediction[0] == 1:
    st.success('The customer is likely to subscribe to a term deposit.')
else:
    st.warning('The customer is not likely to subscribe to a term deposit.')
