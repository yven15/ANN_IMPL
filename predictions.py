import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

model = load_model('model.h5')



#Load encoders & scalers

## load the encoder and scaler
with open('onehot_encoder_geography.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.DataFrame([input_data])

input_data['Gender']=label_encoder_gender.transform(input_data['Gender'])

input_data = input_data.drop(columns='Geography',axis=1)
input_data=pd.concat([input_data,geo_encoded_df],axis=1)

input_scaled = scaler.transform(input_data)

prediction =  model.predict(input_scaled)
print(prediction)



