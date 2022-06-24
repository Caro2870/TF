import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow



plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('time_series.csv', parse_dates=[0], header=None, index_col=0, squeeze=True,
                 names=['fecha', 'unidades'])
df.head()

df.describe()

print(df.index.min())
print(df.index.max())

print(len(df['2017']))
print(len(df['2018']))

meses = df.resample('M').mean()
meses

"""## Visualizaciones"""

plt.plot(meses['2017'].values)
plt.plot(meses['2018'].values)

verano2017 = df['2017-06-01':'2017-09-01']
plt.plot(verano2017.values)
verano2018 = df['2018-06-01':'2018-09-01']
plt.plot(verano2018.values)

"""# Preprocesado de los datos"""

PASOS = 7


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values = values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.head()

"""## Dividimos en set de Entrenamiento y Validación"""

# split into train and test sets
values = reframed.values
n_train_days = 315 + 289 - (30 + PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

"""# Creamos el Modelo de Red Neuronal

## Utilizaremos una Red "normal" Feedforward
"""


def crear_modeloFF():
    model = Sequential()
    model.add(Dense(PASOS, input_shape=(1, PASOS), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["mse"])
    model.summary()
    return model


"""## Entrenamos nuestra máquina"""

EPOCHS = 40

model = crear_modeloFF()

history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=PASOS)

"""## Visualizamos Resultados"""

results = model.predict(x_val)
print(len(results))
plt.scatter(range(len(y_val)), y_val, c='g')
plt.scatter(range(len(results)), results, c='r')
plt.title('validate')
plt.show()

plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.show()

compara = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
compara.columns = ['real', 'prediccion']

inverted = scaler.inverse_transform(compara.values)

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real', 'prediccion']
compara2['diferencia'] = compara2['real'] - compara2['prediccion']
compara2.head()

compara2.describe()

compara2['real'].plot()
compara2['prediccion'].plot()

"""# Predicción

A partir de la última semana de noviembre 2018, intentaremos predecir la primer semana de diciembre.
"""

ultimosDias = df['2018-11-16':'2018-11-30']
ultimosDias

"""## Preparamos los datos para Test"""

values = ultimosDias.values
values = values.astype('float32')
# normalize features
values = values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
reframed.head(7)

values = reframed.values
x_test = values[6:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)
x_test


def agregarNuevoValor(x_test, nuevoValor):
    for i in range(x_test.shape[2] - 1):
        x_test[0][0][i] = x_test[0][0][i + 1]
    x_test[0][0][x_test.shape[2] - 1] = nuevoValor
    return x_test


results = []
for i in range(7):
    parcial = model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test = agregarNuevoValor(x_test, parcial[0])

"""## Re-Convertimos los resultados"""

adimen = [x for x in results]
print(adimen)
inverted = scaler.inverse_transform(adimen)
inverted

"""## Visualizamos el pronóstico"""

prediccion1SemanaDiciembre = pd.DataFrame(inverted)
prediccion1SemanaDiciembre.columns = ['pronostico']
prediccion1SemanaDiciembre.plot()
prediccion1SemanaDiciembre.to_csv('pronostico.csv')

prediccion1SemanaDiciembre

"""# Agregamos el resultado en el dataset"""

i = 0
for fila in prediccion1SemanaDiciembre.pronostico:
    i = i + 1
    ultimosDias.loc['2018-12-0' + str(i) + ' 00:00:00'] = fila
    print(fila)
ultimosDias.tail(14)

"""El artículo completo en www.aprendemachinelearning.com"""
