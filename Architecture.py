import tensorflow as tf
from tensorflow import keras
import numpy as np

class CusConv2D(keras.layers.Layer): #convolution layer (need to detect patterns)
    def __init__(self, filters, kernel_size, stride=1, padding='VALID', activation=None, **kwargs): #initial parameters
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

    def build(self, input_shape): #make variables
        in_channels = input_shape[-1]
        stddev = np.sqrt(2.0 / (self.kernel_size[0] * self.kernel_size[1] * in_channels)) #it ensures that weights are not too small or big

        self.w = self.add_weight( #kernel
            shape=(self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer=tf.initializers.RandomNormal(stddev=stddev),
            trainable=True,
            name='kernel'
        )
        self.b = self.add_weight( #bias
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, x): #Inserting the function into training engine
        z = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        z = tf.nn.bias_add(z, self.b)
        if self.activation:
            return self.activation(z)
        return z


class CusMaxPooling2D(keras.layers.Layer): #decrease the size of the layer, saving important information
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs): #setting the window size
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding

    def call(self, x):
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        return tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding=self.padding) #taking biggest number


class CusFlatten(keras.layers.Layer): # Convert 3D (Height, Width, Channels) into 1D (number of pixels)
    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, [batch_size, -1])


class CusDense(keras.layers.Layer): #create a fully connected layer
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        in_features = input_shape[-1]
        limit = np.sqrt(6 / (in_features + self.units))

        self.w = self.add_weight( #weight
            shape=(in_features, self.units),
            initializer=tf.initializers.RandomUniform(minval=-limit, maxval=limit),
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight( #bias
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, x):
        z = tf.matmul(x, self.w) + self.b #matrix multiplication (x*w+b)
        if self.activation:
            return self.activation(z)
        return z


class CusDropout(keras.layers.Layer): #preventing model from memorizing a dataset
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=False):
        if training:
            return tf.nn.dropout(x, rate=self.rate)
        return x


class CusModel(keras.Model): #the model itself
    def __init__(self, img_height, img_width): #parameters of the images
        super().__init__()

        self.conv1 = CusConv2D(32, (3, 3), activation=tf.nn.relu)
        self.pool1 = CusMaxPooling2D((2, 2))

        self.conv2 = CusConv2D(64, (3, 3), activation=tf.nn.relu)
        self.pool2 = CusMaxPooling2D((2, 2))

        self.conv3 = CusConv2D(128, (3, 3), activation=tf.nn.relu)
        self.pool3 = CusMaxPooling2D((2, 2))


        self.flat = CusFlatten()
        self.dense1 = CusDense(64, activation=tf.nn.relu)
        self.drop = CusDropout(0.2)
        self.out = CusDense(2, activation=tf.nn.softmax)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.drop(x, training=training)
        return self.out(x)