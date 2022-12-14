# ako test skup nije kao trenirajuci
# nacin na koji podelimo skup je los

# podelimo skup na vise skupova (foldove) i recunamo prosek accuracyja

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras import Sequential
from keras.layers import Dense

from sklearn.model_selection import KFold

data = datasets.load_breast_cancer()
ulaz = data.data
izlaz = data.target
ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(ulaz,
                                                                      izlaz,
                                                                      test_size=0.2,
                                                                      shuffle=True,
                                                                      random_state=16)

# skaliranje ulaznih podataka
scaler = StandardScaler().fit(ulaz_trening)
ulaz_trening_norm = scaler.transform(ulaz_trening)
# samo jednom fitujemo
ulaz_test_norm = scaler.transform(ulaz_test)

def make_model(n_in, n_out):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=n_in))
    model.add(Dense(n_out, activation='sigmoid'))

    # metrics je nesto sto samo prati mrezu -> ne utice na rez
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

kf = KFold(n_splits=5, shuffle=True, random_state=12)
A_svi = []
A_max = 0
for trening, val in kf.split(ulaz_trening_norm, izlaz_trening):
    model = make_model(ulaz_trening_norm.shape[1], 1)
    history = model.fit(ulaz_trening_norm[trening, :], izlaz_trening[trening],
                        epochs=20,
                        batch_size=16,
                        verbose=0)
    A = model.evaluate(ulaz_trening_norm[val, :], izlaz_trening[val])[1]
    print(A)
    A_svi.append(A)
    if A > A_max:
        best_model = model
        A_max = A

print("---------------------------------")
print(np.mean(A_svi))
print(best_model.evaluate(ulaz_test_norm, izlaz_test)[1])