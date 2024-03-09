README file:

```markdown
# Neural Network Architectures with TensorFlow and Keras

This README provides explanations for two neural network architectures implemented using TensorFlow and Keras.

## Model 1

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

This model consists of three dense layers with ReLU activation functions, followed by batch normalization layers after each dense layer. The first layer flattens the input of shape (28, 28), which is typically used for images in this format. Batch normalization layers help stabilize and speed up the training process by normalizing the inputs of each layer. The last layer uses a softmax activation function, suitable for multi-class classification tasks.

## Model 2

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

This model is similar to the first one but incorporates activations directly after each dense layer, rather than using the activation parameter within the `Dense` layer. Also, the `use_bias=False` argument is used in the dense layers, implying that bias terms are not used in those layers.

## Viewing Variables in a Layer

```python
import tensorflow as tf

# List variables within the second layer of the model along with their training status
[(var.name,var.trainable)for var in model.layers[1].variables]
```

This code snippet lists the variables within the second layer of the model along with their training status (whether they are trainable or not).

---

These models are suitable for tasks like digit recognition on the MNIST dataset.
```

Remember to adjust the formatting and wording according to your specific needs and preferences.
