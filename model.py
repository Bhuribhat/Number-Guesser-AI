import math
import tensorflow as tf
import tensorflow_addons as tfa


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0


# rotate (-50, 50) degree and add some noise
def augment_data(image, label):
    angle = tf.random.uniform(shape=[], minval=-50, maxval=50, dtype=tf.float32)
    image = tfa.image.rotate(image, angle * math.pi / 180, interpolation='BILINEAR')
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=90, max_jpeg_quality=100)
    return image, label


# Use the data augmentation function to modify the training images
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(augment_data)

# Build and train the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset.batch(64), epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on the test data (98.5 %)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

# Save model
model.save('./assets/number_model.h5')