from math import sin

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv('./cleared_t_i_v.csv')
data = df.values

R = 1000
L = 0.001

nn = keras.Sequential([
    keras.layers.Dense(5, activation='tanh', input_shape=(1,)),
    keras.layers.Dense(5, activation='tanh'),
    keras.layers.Dense(1, activation='tanh')  # i(t)
    # keras.layers.Dense(1, activation='custom') f(t, v)
])
nn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

t = tf.constant(1.0, shape=(1, 1))

# https://www.tensorflow.org/tutorials/customization/autodiff
# https://www.tensorflow.org/api_docs/python/tf/gradients
with tf.GradientTape() as gt:
  gt.watch(t)
  z = 2*t
  # z = nn.predict(t, steps=1)

# Derivative of z with respect to the original input tensor x
dz_dx = gt.gradient(z, t)

print(dz_dx)


def v_in(t):
    return sin(100*t)


def i(t):
    return nn.predict([t])


def di_dt(t):
    return t


def f(t, v):
    return di_dt(t) + (R/L)*i(t) - (1/L)*v(t)
