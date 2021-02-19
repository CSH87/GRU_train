import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
dataset = pd.read_csv('TrainDataSetOfAll2.csv') 
test_dataset = pd.read_csv('TestDataSetOfAll2.csv')
normalize_Layer = tf.keras.layers.LayerNormalization(axis=1)
test_data_X = np.array(test_dataset.iloc[:, 5:9].values)
#test_data_X = normalize_Layer(test_data_X).numpy()
test_data_X = test_data_X.reshape(-1,10,4)
i=0
test_data_Y = []
for label in test_dataset.iloc[:, -1].values:
    if i%10==0:
        test_data_Y.append(label)
    i=i+1
test_data_Y=np.array(test_data_Y, dtype=int)

i=0
k=0
train_data_Y=[]
train_data_X = np.array(dataset.iloc[:, 5:9].values)
#train_data_X = normalize_Layer(train_data_X).numpy()
train_data_X = train_data_X.reshape(-1,10,4) 
for label in dataset.iloc[:, -1].values:
    if i%10 ==0:
        train_data_Y.append(label)
      
    i=i+1
train_data_Y=np.array(train_data_Y, dtype=int)


#"""
model = keras.Sequential()
model.add(layers.GRU(128, input_shape = (10,4), return_sequences= True))
#model.add(layers.Dropout(0.2))
model.add(layers.GRU(64, return_sequences = False))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(64 , activation='sigmoid'))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(units = 1, activation ='sigmoid'))
model.summary()
#opt = keras.optimizers.Adam(learning_rate=0.006)
model.compile(loss="poisson", optimizer = "adam", metrics= ['accuracy'])

myCheckPoint = ModelCheckpoint('my_model.h5', 'val_accuracy', save_best_only= True, verbose=1)
train_history = model.fit(train_data_X, train_data_Y, batch_size= 32, epochs=100, validation_split = 0.2, verbose=1, callbacks=[myCheckPoint])
#"""
#model = keras.models.load_model('acc7192.h5')
model2 = load_model('my_model.h5')
print('model 1 : \n')
model.evaluate(test_data_X, test_data_Y)
print('model 2 : \n')
model2.evaluate(test_data_X, test_data_Y)

print('end')