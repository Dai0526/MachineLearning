"""
Tutorials from https://www.tensorflow.org/guide/basics
"""

import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(x + x)
print(5 * x)
print(x @ tf.transpose(x))
print(tf.concat([x, x, x], axis=0))
print(tf.nn.softmax(x, axis=-1))
print(tf.reduce_sum(x))

## Training loops
import matplotlib
from matplotlib import pyplot as plt

print("Start Training Loops\r\n")

#create example data - a cloud of points that loosely follows a quadratic curve:

matplotlib.rcParams['figure.figsize'] = [9, 6]
x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

def f(x):
  y = x**2 + 2*x - 5
  return y

y = f(x) + tf.random.normal(shape=[201])

# plt.plot(x.numpy(), y.numpy(), '.', label='Data')
# plt.plot(x, f(x),  label='Ground truth')
# plt.show()

# create model
class Model(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = units, activation = tf.nn.relu, kernel_initializer = tf.random.normal, bias_initializer = tf.random.normal)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training = True):
        x = x[:, tf.newaxis]
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.squeeze(x, axis = 1)

model = Model(64)

plt.plot(x.numpy(), y.numpy(), '.', label='data')
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Untrained predictions')
plt.title('Before training')
plt.show()

# Train
variables = model.variables
optimizer = tf.optimizers.SGD(learning_rate = 0.01)
for step in range(1000):
    with tf.GradientTape() as tape:
        prediction = model(x)
        error = (y-prediction)**2
        mean_error = tf.reduce_mean(error)
    gradient = tape.gradient(mean_error, variables)
    optimizer.apply_gradients(zip(gradient, variables))
    if step % 100 == 0:
        print(f'Mean squared error: {mean_error.numpy():0.3f}')

plt.plot(x.numpy(),y.numpy(), '.', label="data")
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Trained predictions')
plt.title('After training')
plt.show()

# train using keras default built in methods
new_model = Model(64)
new_model.compile(loss = tf.keras.losses.MSE, optimizer = tf.optimizers.SGD(learning_rate = 0.01))
history = new_model.fit(x, y, epochs = 100, batch_size = 32, verbose = 0)

# plot trainning history
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylim([0, max(plt.ylim())])
plt.ylabel('Loss [Mean Squared Error]')
plt.title('Keras training progress')
plt.show()

# save new trained model
model.save('./my_model')