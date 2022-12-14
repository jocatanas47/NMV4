import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
# from keras.regularizers import l1
from keras.regularizers import l2

pod = pd.read_csv('podaciCas04.csv', header=None)
# baza nema imena kolona pa ih dodajemo
pod.columns = ['x1', 'x2', 'd']

ulaz = pod.iloc[:, :2].to_numpy()
izlaz = pod.d.to_numpy()

klase = np.unique(izlaz)
K0 = ulaz[izlaz == klase[0], :]
K1 = ulaz[izlaz == klase[1], :]
K2 = ulaz[izlaz == klase[2], :]

plt.figure()
plt.plot(K0[:, 0], K0[:, 1], "o")
plt.plot(K1[:, 0], K1[:, 1], "*")
plt.plot(K2[:, 0], K2[:, 1], "d")
plt.show()

izlaz_cat = to_categorical(izlaz)
ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(ulaz,
                                                                      izlaz_cat,
                                                                      test_size=0.2,
                                                                      shuffle=True,
                                                                      random_state=20)
ulaz_ob, ulaz_val, izlaz_ob, izlaz_val = train_test_split(ulaz_trening,
                                                          izlaz_trening,
                                                          test_size=0.2,
                                                          shuffle=True,
                                                          random_state=25)

model = Sequential()
model.add(Dense(200, activation='relu',
                kernel_regularizer=l2(0.1), input_dim=ulaz.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(izlaz_cat.shape[1], activation='softmax'))

model.compile('adam', loss='categorical_crossentropy')

history = model.fit(ulaz_ob, izlaz_ob,
                    epochs=1000,
                    batch_size=128,
                    validation_data=(ulaz_val, izlaz_val),
                    verbose=0)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

pred_trening = model.predict(ulaz_trening)
pred_trening = np.argmax(pred_trening, axis=1)
izlaz_trening_klase = np.argmax(izlaz_trening, axis=1)
print(accuracy_score(izlaz_trening_klase, pred_trening))

pred_test = model.predict(ulaz_test)
pred_test = np.argmax(pred_test, axis=1)
izlaz_test_klase = np.argmax(izlaz_test, axis=1)
print(accuracy_score(izlaz_test_klase, pred_test))

Ntest = 500
x1test = np.linspace(-4, 7, Ntest)
x2test = np.linspace(-4, 7, Ntest)

x1grid, x2grid = np.meshgrid(x1test, x2test)
x1grid = x1grid.reshape([1, Ntest**2]).T
x2grid = x2grid.reshape([1, Ntest**2]).T

grid = np.append(x1grid, x2grid, axis=1)

pred_grid = model.predict(grid)
pred_grid = np.argmax(pred_grid, axis=1)

K0pred = grid[pred_grid == klase[0], :]
K1pred = grid[pred_grid == klase[1], :]
K2pred = grid[pred_grid == klase[2], :]

plt.figure()
plt.plot(K0pred[:, 0], K0pred[:, 1], 'r.')
plt.plot(K1pred[:, 0], K1pred[:, 1], 'b.')
plt.plot(K2pred[:, 0], K2pred[:, 1], 'g.')
plt.show()