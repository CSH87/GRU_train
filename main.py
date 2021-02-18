import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
dataset = pd.read_csv('TrainDataSetOfAll.csv') 
test_dataset = pd.read_csv('TestDataSetOfAll.csv')
normalize_Layer = tf.keras.layers.LayerNormalization(axis=1)
test_data_X = np.array(test_dataset.iloc[:, 5:9].values)
test_data_X = normalize_Layer(test_data_X).numpy()
#test_data_X = test_data_X / np.linalg.norm(test_data_X)
test_data_X = test_data_X.reshape(-1,10,4)
i=0
test_data_Y = []
for label in test_dataset.iloc[:, -1].values:
    if i%10==0:
        test_data_Y.append(label)
    i=i+1
test_data_Y=np.array(test_data_Y, dtype=int)
"""
for row in csvfile:
    print(row)
"""
i=0
k=0
train_data_Y=[]
train_data_X = np.array(dataset.iloc[:, 5:9].values)
train_data_X = normalize_Layer(train_data_X).numpy()
#train_data_X = train_data_X / np.linalg.norm(train_data_X)
train_data_X = train_data_X.reshape(-1,10,4) 
for label in dataset.iloc[:, -1].values:
    if i%10 ==0:
        train_data_Y.append(label)
      
    i=i+1
train_data_Y=np.array(train_data_Y, dtype=int)



model = keras.Sequential()
model.add(layers.GRU(128, input_shape = (10,4), return_sequences= True))
model.add(layers.Dropout(0.2))
model.add(layers.GRU(64, return_sequences = False))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units = 1, activation ='sigmoid'))
model.summary()
opt = keras.optimizers.Adam(learning_rate=0.002)
model.compile(loss="binary_crossentropy", optimizer = opt, metrics= ['accuracy'])


train_history = model.fit(train_data_X, train_data_Y,batch_size= 32, epochs=100, validation_split = 0.2, verbose=1)

model.evaluate(test_data_X, test_data_Y)
print('end')