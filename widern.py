from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

ch_axis = -1

def conv(filt, smpl, strid=1):
    return Conv2D(filt, smpl, strides=strid, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), use_bias=False)
def BN(x):
    return BatchNormalization(axis=ch_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
def relu(x):
    return Activation('relu')(x)

def init_conv(init):
    x = conv(16, 3)(init)
    x = BN(x)
    x = relu(x)
    return x

def first_residual(init, filt, strid):
    x = conv(filt, 3, strid)(init)
    x = BN(x)
    x = relu(x)
    x = conv(filt, 3)(x)
    shortcut = conv(filt, 1, strid)(init)
    a = Add()([x, shortcut])
    return a

def residual_unit(init, filt, dropout):
    x = BN(init)
    x = relu(x)
    x = conv(filt, 3)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = BN(x)
    x = relu(x)
    x = conv(filt, 3)(x)    
    a = Add()([init, x])
    return a

def residual_block(init, filt, strid, N, dropout):
    x = first_residual(init, filt, strid)
    for i in range(N - 1):
        x = residual_unit(x, filt, dropout)
    x = BN(x)
    x = relu(x)
    return x
    
def create_wrn(init, cl, N, k, dropout):
    inp = Input(shape=init)
    x = init_conv(inp)
    x = residual_block(x, 16*k, 1, N, dropout)
    x = residual_block(x, 32*k, 2, N, dropout)
    x = residual_block(x, 64*k, 2, N, dropout)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(cl, activation='softmax')(x)
    model = Model(inp, x)
    return model
