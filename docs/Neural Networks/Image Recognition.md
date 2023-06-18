# Image Recognition

## Building the Neural Network Model

Multiple types models can be be built, 

* `Sequential`: Linear stack of layers, each layer has exactly one input tensor and one output tensor. Simple and straightforward 

* `Functional API`: More flexibility in creating models with multiple inputs, multiple outputs, shared layers, and complex network architectures. Suitable for building models with branching or merging architectures, as well as models with multiple inputs or outputs.

* `Model`: The `Model` class is a more general and flexible way.Define your own custom models by subclassing the `Model` class and implementing the forward pass logic in the `call` method. This class provides full control over the model's architecture and is commonly used for advanced customization or when building complex models.

```python
my_model = tf.keras.Sequential([
    Flatten(),
    Dense(10, activation="relu"), # Add 10 nodes each implementing Relu Algorithm
    Dense(10, activation="relu"), 
    Dense(4, activation="softmax") # Add 4 nodes each implementing softmax
])
```

In the above case 4 nodes are added to the last layer as we are trying to classify the image into one of the four categories. `Softmax` determines the probability of the given image matching each shape.

## Compiling the model

Number of epochs is the number of times the model is trained. Check out how to [[persist models|Model Persistence]].

```python
my_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),  
				# Loss function that we want can be any from the class 
                optimizer=tf.keras.optimizers.Adam(),  
                # Similar to Gradient Descent
                metrics=["accuracy"]) # Metrics to be measured

model_history = my_model.fit(train_data, epochs=5, validation_data=validation_data)
```


## Plot the Loss Metrics Curve

```python
pd.DataFrame(model_history.history).plot()
```

## Testing the Model

Process the image to get the matrix. The image data must be similar to model training data with respect to color, alpha and other such parameters

```python
def preproc_img(path):
  img = tf.io.read_file(path) #decode_img reads only bin
  img = tf.io.decode_image(img) # Converts image to 3D matrix of numbers
  if img.shape[-1] == 1:
	# Convert to RGB, file is in greyscale, since model trained on RGB
    img = tf.image.grayscale_to_rgb(img)
  img = tf.image.resize(img, [256, 256])/255. # Resize and rescale
  return img
```

## Testing

```python
import numpy as np

img = preproc_img("./test_data/1000.png")
model_output = my_model.predict(tf.expand_dims(img, 0))

print(model_output) 
# Will return 4 numbers, each the probability of the input falling to the category
np.argmax(model_output) # Get the maximum one
```