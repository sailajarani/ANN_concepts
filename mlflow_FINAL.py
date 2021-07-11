import keras.regularizers
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import mlflow
import time
import os

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# create a validation data set from the full training data
# Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# scale the test set as well
X_test = X_test / 255.
mlflow.tensorflow.autolog()

# model with relu activation
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)

# model with sigmoid activation function
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='sigmoid', kernel_initializer='glorot_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='sigmoid',kernel_initializer='glorot_normal', name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_2 = tf.keras.models.Sequential(layers=LAYERS)

    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)
    model_2.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

    history = model_2.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid))

# model with leakyrelu activation function
with mlflow.start_run():

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='LeakyReLU', kernel_initializer='he_normal',name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='LeakyReLU',kernel_initializer='he_normal', name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_3 = tf.keras.models.Sequential(layers=LAYERS)

    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)
    model_3.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

    history = model_3.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid))
# model with elu activation function
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
             tf.keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal', name='hiddenLayer1'),
             tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal',name='hiddenLayer2'),
             tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_4 = tf.keras.models.Sequential(layers=LAYERS)

    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)
    model_4.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

    history = model_3.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid))

# model with selu activation with lecun kernal initlizer / weight initilizer
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='selu', kernel_initializer='lecun_normal',name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal' ,name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_5 = tf.keras.models.Sequential(layers=LAYERS)
    #print(model_5.summary())
    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    model_5.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model_5.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model_5.predict(X_test)
    preds = np.round(preds)
# model with batch normalization
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(300, activation='relu', name='hiddenLayer1'),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(100, activation='relu', name='hiddenLayer2'),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_6 = tf.keras.models.Sequential(layers=LAYERS)
    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']

    model_6.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model_6.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model_6.predict(X_test)
    preds = np.round(preds)

with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.SGD(lr=0.01,name='SGD')
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.SGD(lr=0.01, name='nestervo',nesterov=True)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.SGD(lr=0.01, name='momentum',momentum=0.9)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)

with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.Adagrad(lr=0.01, name='adagard',initial_accumulator_value=0.1,epsilon=1e-07)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.Adam(lr=0.01, name='adam',beta_1=0.9,beta_2=0.99)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1(0.1), name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.Adam(lr=0.01, name='adam',beta_1=0.9,beta_2=0.99)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)

with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01), name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.Adam(lr=0.01, name='adam',beta_1=0.9,beta_2=0.99)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(300, activation='relu',kernel_initializer='he_normal', name='hiddenLayer1'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal',name='hiddenLayer2'),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model1 = tf.keras.models.Sequential(layers=LAYERS)
    print(model1.summary())
    LOSS = 'sparse_categorical_crossentropy'
    METRICS = ['accuracy']

    model1.compile(loss=LOSS, metrics=METRICS)
    tf.keras.optimizers.Adam(lr=0.01, name='adam',beta_1=0.9,beta_2=0.99)
    EPOCHS = 30
    VALIDATION_DATA = (X_valid, y_valid)

    model1.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION_DATA)
    preds = model1.predict(X_test)
    preds = np.round(preds)


# model with callbacks
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name='inputLayer'),
              tf.keras.layers.Dense(300, activation='relu', name='hiddenLayer1'),
              tf.keras.layers.Dense(100, activation='relu', name='hiddenLayer2'),
              tf.keras.layers.Dense(10, activation='softmax', name='outputLayer')]
    model_7 = tf.keras.models.Sequential(layers=LAYERS)

    LOSS = 'sparse_categorical_crossentropy'
    OPTIMIZER = 'SGD'
    METRICS = ['accuracy']
    VALIDATION_DATA = (X_valid, y_valid)
    # LOSS = "mse"
    # OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)

    model_7.compile(loss=LOSS, optimizer=OPTIMIZER)

    EPOCHS = 100


    def get_log_path(log_dir="logs/fit"):
        fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
        log_path = os.path.join(log_dir, fileName)
        print(f"saving logs at: {log_path}")
        return log_path


    log_dir = get_log_path()

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs")

    CALLBACKS = [checkpoint_cb, early_stopping_cb, tb_cb]

    history = model_7.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid), callbacks=CALLBACKS)

