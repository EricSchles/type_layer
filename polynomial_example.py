from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class Polynomial(tf.Module):
    def __init__(self, in_features, out_features, break_points, polynomials, init_weights=None, name=None):
        super().__init__(name=name)
        if init_weights is None:
            self.init_weights = tf.random.normal([out_features, out_features])
        else:
            self.init_weights = init_weights
        self.w = tf.Variable(
            self.init_weights, name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
        #break points MUST be sorted lowest to highest
        self.break_points = break_points
        self.polynomials = polynomials

    #instead of "manual breakpoints, consider quartiles or some other finer grained split of the data
    def polynomial(self, x):
        if x < break_points[0]:
            return 0
        
        for index in range(len(self.break_points)-1):
            next_index = index + 1
            if x > self.break_points[index] and x < self. break_points[next_index]:
                return polynomials[index](x)
        else:
            return polynomials[-1](x)
        
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        transformed_rows = [tf.map_fn(self.polynomial, x.numpy()[i]) for i in range(x.shape[0])]
        return tf.stack(transformed_rows, axis=1)
        
class NeuralNet(Model):
    def __init__(self, X_in, X_out, optimizer, break_points, polynomials):
        super(NeuralNet, self).__init__()
        self.layer = Polynomial(X_in, X_out, break_points, polynomials)
        self.optimizer = optimizer
        
    def call(self, x):
        return self.layer(x)

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape() as tape:
            pred = self.call(x)
            loss = mse(pred, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

def mse(x, y):
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    return tf.metrics.MSE(x, y)

def polynomial_generator(n):
    return lambda x: sum([x**i for i in range(1, n+1)])

if __name__ == '__main__':
    X = pd.read_csv("X.csv")
    y = np.load("y.npy")
    X = X.values
    X_val = pd.read_csv("X_val.csv")
    X_val = X_val.values
    y_val = np.load("y_val.npy")
    learning_rate = 0.9
    optimizer = tf.optimizers.Adam(learning_rate)
    break_points = [0, 2, 4, 6, 8]
    polynomials = [polynomial_generator(i) for i in range(len(break_points))]
    nn = NeuralNet(X.shape[0], X.shape[1], optimizer, break_points, polynomials)
    num_steps = 1
    for step in range(num_steps):
        nn.step(X, y)
        pred = nn(X_val)
        loss = mse(pred, y_val)
    print("mse", loss)
