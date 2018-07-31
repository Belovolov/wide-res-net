import numpy    as np
import widern   as wrn
import os

from keras.datasets             import cifar100
from keras.callbacks            import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.preprocessing.image  import ImageDataGenerator
from keras.models               import model_from_json
from keras.optimizers           import SGD
from keras.utils                import np_utils

"""
Wide Residual Network configuration

(N x 6) + 4 : The depth of the network
k           : The width of the network

Example:
    For WRN-28-10 : N = 4, k = 10
    For WRN-22-8  : N = 3, k = 8
    For WRN-16-8  : N = 2, k = 8

"""
N = 3
k = 8

classes = 100
nb_epoch = 150
lrstep = [45, 90, 120]
batch_size = 128

init_shape = (32, 32, 3)
file_model = 'CIFAR100-WRN-{}-{}' .format(((N*6)+4), k)

""" End of configuration """



def lrschedule(epoch_idx):
    if (epoch_idx + 1) < lrstep[0]:
        return 0.1
    elif (epoch_idx + 1) < lrstep[1]:
        return 0.02
    elif (epoch_idx + 1) < lrstep[2]:
        return 0.004
    return 0.0008

if not os.path.exists('./results'):
    os.makedirs('./results')

(trainX, trainY), (testX, testY) = cifar100.load_data()
trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)
generator.fit(trainX, seed=0, augment=True)

model = wrn.create_wrn(init_shape, classes, N, k, dropout=0.3)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['acc'])
print('\n===== Training a Wide Residual Network of {} depth and {} width using CIFAR-100 dataset for {} epochs =====\n' .format(((N*6)+4), k, nb_epoch))

model_json = model.to_json()
with open(file_model + '.json', 'w') as json_file:
    json_file.write(model_json)

callbacks = [ LearningRateScheduler(schedule=lrschedule), ModelCheckpoint('results/CIFAR100.weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')]

results = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, epochs=nb_epoch,
                   callbacks=callbacks,
                   validation_data=(testX, testY),
                   validation_steps=testX.shape[0] // batch_size,)

model.save_weights(file_model + '.h5')
print('Saved model to disk')


