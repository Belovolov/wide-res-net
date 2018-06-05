import numpy as np
import pickle
import sklearn.metrics as metrics

import wide_residual_network as wrn
from keras.datasets import cifar100
from funcy                     import concat, identity, juxt, partial, rcompose, repeat, repeatedly, take
from keras.callbacks           import LearningRateScheduler, ModelCheckpoint, TensorBoard
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import Model, save_model, model_from_json
from keras.optimizers import SGD
from operator                  import getitem

from keras import backend as K

import matplotlib.pyplot as plt


batch_size = 128
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)


model = wrn.create_wide_residual_network(init_shape, nb_classes=100, N=4, k=10, dropout=0.3)

model.summary()


model.compile(loss="categorical_crossentropy", optimizer=SGD(momentum=0.9), metrics=["acc"])

model_json = model.to_json()
with open("100wrn-28-10.json", "w") as json_file:
    json_file.write(model_json)

callbacks = [ LearningRateScheduler(partial(getitem, tuple(take(nb_epoch, concat(repeat(0.1, 60), repeat(0.02, 60), repeat(0.004, 40), repeat(0.0008)))))),
          ModelCheckpoint('results/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
          TensorBoard(log_dir='./logs', batch_size=32, write_graph=True, write_grads=False, write_images=True)]


results = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, epochs=nb_epoch,
                   callbacks=callbacks,
                   validation_data=(testX, testY),
                   validation_steps=testX.shape[0] // batch_size,)

model.save_weights("100wrn-28-10.h5")
save_model(model, './results/model.h5')
print("Saved model to disk")
with open('./results/history.pickle', 'wb') as f:
    pickle.dump(results.history, f)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yPred = kutils.to_categorical(yPred)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
##
##print(results.history.keys())
### summarize history for accuracy
##plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
##
