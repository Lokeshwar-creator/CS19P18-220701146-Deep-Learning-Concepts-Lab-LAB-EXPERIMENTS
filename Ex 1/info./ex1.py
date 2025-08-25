 # Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Step 1: Load and Preprocess Data
# ------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images (0–255 -> 0–1 range)
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

print("Training data:", x_train.shape, "Labels:", y_train.shape)
print("Testing data:", x_test.shape, "Labels:", y_test.shape)

# ------------------------------
# Step 2: Build Model
# ------------------------------
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Flatten image to 1D
    layers.Dense(512, activation='relu'),          # Hidden layer
    layers.Dropout(0.2),                           # Dropout to avoid overfitting
    layers.Dense(10, activation='softmax')         # Output layer (10 digits)
])

# ------------------------------
# Step 3: Compile Model
# ------------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# Step 4: Train Model
# ------------------------------
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=320,
    verbose=1
)

# ------------------------------
# Step 5: Evaluate Model
# ------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# ------------------------------
# Step 6: Make Predictions
# ------------------------------
predictions = model.predict(x_test)

# Show sample predictions
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nAct: {np.argmax(y_test[i])}")
    plt.axis("off")
plt.show()
