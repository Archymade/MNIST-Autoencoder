import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
import tensorflow_probability as tfp
#from deepcopy import copy


class AutoEncoder(Model):
    """ Class blueprint for Autoencoder. """

    def __init__(self, optimizer, in_dims, latent_dims, encoder_name=None,
                 decoder_name=None, variational=False):

        super().__init__()

        self.latent_dims = latent_dims
        self.in_dims = in_dims

        self.stochastic = variational
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        self.encoder = self.generate_encoder()
        self.decoder = self.generate_decoder()

        self.optimizer = optimizer

        # self.encoder_optimizer = copy(optimizer)
        # self.decoder_optimizer = copy(optimizer)

        if self.stochastic:
            self.distribution_layer = layers.Dense(units=2 * self.latent_dims)
            self.var_mean, self.var_logvar = None, None

        self.encoder.compile(loss='binary_crossentropy', optimizer=self.encoder_optimizer)
        self.decoder.compile(loss='binary_crossentopy', optimizer=self.decoder_optimizer)

    def generate_encoder(self):
        model = Sequential([layers.Input(shape=self.in_dims),
                            # tf.keras.layers.experimental.preprocessing.Rescaling(1/255),
                            layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=1),
                            layers.BatchNormalization(),
                            layers.LeakyReLU(),

                            layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=2),
                            layers.BatchNormalization(),
                            layers.LeakyReLU(),

                            layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=2),
                            layers.BatchNormalization(),
                            layers.LeakyReLU(),

                            layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=1),
                            layers.BatchNormalization(),
                            layers.LeakyReLU(),

                            layers.Flatten(),
                            layers.Dense(units=self.latent_dims)
                            ], name=self.encoder_name if self.encoder_name else None)

        return model

    def generate_decoder(self):
        model = Sequential(name=self.decoder_name if self.decoder_name else None)

        model.add(layers.Input(shape=[self.latent_dims]))
        model.add(layers.Dense(units=7 * 7 * 16, activation='relu'))
        model.add(layers.Reshape(target_shape=[7, 7, 16]))

        model.add(layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(filters=self.in_dims[-1], kernel_size=3, strides=1, padding='valid',
                                activation='sigmoid'))

        return model

    def encode(self, x):
        z_ = self.encoder(x)
        return z_

    def distribute(self, x):
        x = self.distribution_layer(x)
        self.var_mean, self.var_logvar = tf.split(x, num_or_size_of_split=2, axis=1)

        return self.var_mean, self.var_logvar

    def reparametrize(self, x):
        m, l = self.distribute(x)
        dist = tfp.python.distributions.Normal(loc=0, scale=1)
        epsilon = dist.sample(m.shape)

        return m + epsilon * tf.exp(0.5 * l)

    def decode(self, x):
        sample = self.decoder(x)
        return sample

    def call(self, x):
        z_ = self.encode(x)
        if self.stochastic:
            z = self.reparametrize(z_)

        y = self.decode(z_)

        return y
