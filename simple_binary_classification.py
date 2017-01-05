import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

print dataset.shape

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

split_at = 180
x_train = X[0:split_at]
x_val = X[split_at:]
y_train = encoded_Y[0:split_at]
y_val = encoded_Y[split_at:]

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
model = create_baseline()
model.fit(x_train, y_train, nb_epoch=50, validation_data=(x_val, y_val))
