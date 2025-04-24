import pandas as pd
from jinja2.environment import load_extensions
from keras.src.optimizers.optimizer import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

data = pd.read_csv("Churn_Modelling.csv")
#print(data.head())

## Text preprocessing
# Removing unnecessary columns in the table
data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)

#print(data.head())


#Encode Categorical Variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

#print(data['Gender'])

#but for Geography we can't use 0 or 1 right so one hot encoder we use it
onehot_encoder_geography = OneHotEncoder()
geo_encoder_sparseMatrix=  onehot_encoder_geography.fit_transform(data[['Geography']])

#print(onehot_encoder_geography.get_feature_names_out())
geo_encoder_dense = geo_encoder_sparseMatrix.toarray()
geo_encoded_df = pd.DataFrame(geo_encoder_dense,columns=onehot_encoder_geography.get_feature_names_out())
#print(geo_encoded_df)

#Lot of stuff happened
# 1. Converted into 2D representation for Geography colmun
# 2. Then removed the old text column and replaced it with new 2D column
# 3.Check the recording for better understanding

data = data.drop('Geography',axis=1)
data = pd.concat([data,geo_encoded_df],axis=1)
#print(data.head())

#Pickle comes here for storing labelEncoding,oneHotEncoding Object instances

# with open('label_encoder_gender.pkl','wb') as file:
#     pickle.dump(label_encoder_gender,file)
#
# with open('onehot_encoder_geography.pkl','wb') as file:
#     pickle.dump(onehot_encoder_geography,file)

#Divide dataSet into dependent & Independent one
X=data.drop('Exited',axis=1)
y=data['Exited']

#print(X.shape)

#Split data into Training & Testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# print(X_train.shape[1])
# print(X_test)


# with open('scaler.pkl','wb') as file:
#     pickle.dump(scaler,file)

#Now until here we did data processing and created train & test data


#ANN IMPLEMENTAION

#Necessary libraries

import tensorflow as tf
import tensorboard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

#Build ANN Model
model = Sequential([
  Dense(64,activation='relu',input_shape=(X_train.shape[1],)),  #HL1 Connected with input layer
  Dense(32,activation='relu'),  #HL2
  Dense(1,activation='sigmoid')
])

print(model.summary())

#Compile the model
#If values provided here in compile() function then these are static values
#Its like otpimizer = "Adam" i.e Adam has default 0.1 learning rate but if you want
#to control this !!
#Customized for optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])


#Setup the Tensor board for log visuals
log_dir = "logs/fit"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)

#setup the Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

#Train the model
history = model.fit(
    X_train,y_train,validation_data=(X_test,y_test),
    epochs=100,callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5')


