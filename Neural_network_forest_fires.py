import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
forestfires = pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/forestfires.csv")
forestfires.head()
#As dummy variables are already created, we will remove the month and alsoday columns
forestfires.drop(["month","day"],axis=1,inplace = True)
forestfires.head()
forestfires["size_category"].value_counts()
forestfires.isnull().sum()
forestfires.describe()
##I am taking small as 0 and large as 1
forestfires.loc[forestfires["size_category"]=='small','size_category']=0
forestfires.loc[forestfires["size_category"]=='large','size_category']=1
forestfires["size_category"].value_counts()
# natural logarithm scaling (+1 to prevent errors at 0)
forestfires.loc[:, ['rain', 'area']] = forestfires.loc[:, ['rain', 'area']].apply(lambda x: np.log(x + 1), axis = 1)
# visualizing
fig, ax = plt.subplots(2, figsize = (5, 8))
ax[0].hist(forestfires['rain'])
ax[0].title.set_text('histogram of rain')
ax[1].hist(forestfires['area'])
ax[1].title.set_text('histogram of area')
#Train Test Split
import tensorflow as tf
X = forestfires.iloc[:,0:28]
y = forestfires.iloc[:,28]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train=tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train=tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test=tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test=tf.convert_to_tensor(y_test, dtype=tf.float32)
#Feature Scaling: StandardScaler
# fitting scaler
sc_features = StandardScaler()
# transforming features
X_test = sc_features.fit_transform(X_test)
X_train = sc_features.transform(X_train)
# features
X_test = pd.DataFrame(X_test, columns = X.columns)
X_train = pd.DataFrame(X_train, columns = X.columns)
# labels
y_test = pd.DataFrame(y_test, columns = ['size_category'])
y_train = pd.DataFrame(y_train, columns = ['size_category'])
X_train.head()
#Neural Network
model = Sequential()
# input layer + 1st hidden layer
model.add(Dense(6, input_dim=28, activation='relu'))
# 2nd hidden layer
model.add(Dense(6, activation='relu'))
# output layer
import tensorflow
example_model = tensorflow.keras.Sequential()
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Activation = tensorflow.keras.layers.Activation
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense

model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))
model.summary()
# Compile Model
model.compile(optimizer = 'adam', metrics=['accuracy'], loss ='binary_crossentropy')
# Train Model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 10, epochs = 100)
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, valid_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))
plt.figure(figsize=[8,5])
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.legend()
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves Epoch 100, Batch Size 10', fontsize=16)
plt.show()