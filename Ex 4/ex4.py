# EX NO: 4 TRANSFER LEARNING WITH CNN AND VISUALIZATION
# Aim: To build a convolutional neural network with transfer learning and perform visualization

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Download and load the dataset (CIFAR-10)
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# 2. Preprocess: normalize and convert labels
x_train = preprocess_input(x_train.astype('float32'))
x_test = preprocess_input(x_test.astype('float32'))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Load pretrained MobileNetV2 (without top layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False   # freeze pretrained layers

# Add custom classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test), batch_size=64)

# 6. Evaluate performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# 7. Visualization: Training vs Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# 8. Predict some test images
preds = model.predict(x_test[:5])
print("Predicted Labels:", np.argmax(preds, axis=1))
print("True Labels:", np.argmax(y_test[:5], axis=1))

# Show first 5 test images with predictions
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow((x_test[i]+1)/2)   # rescale for display
    plt.title(class_names[np.argmax(preds[i])])
    plt.axis("off")
plt.show()
