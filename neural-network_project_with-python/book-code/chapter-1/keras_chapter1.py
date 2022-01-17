from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

np.random.seed(9)

model = Sequential()

# layer 1
model.add(Dense(unit = 4, activation = 'sigmoid', input_dim = 3))

#output layer
model.add(Dense(units = 1, activation = 'sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(lr = 1)
model.compile(loss = 'mean_squarred_erros', optimizers = sgd)

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],[1],[1],[0]])

model.fit(x , y, epochs = 1500, verbose = False)

print(model.predict(x))