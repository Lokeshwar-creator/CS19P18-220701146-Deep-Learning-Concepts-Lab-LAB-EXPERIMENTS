# EX NO: 3 IMAGE CLASSIFICATION ON CIFAR-10 DATASET USING CNN

# -----------------------------
# Step 1: Import Libraries
# -----------------------------
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Step 2: Load and Visualize Data
# -----------------------------
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# CIFAR-10 class labels
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Show sample images
def show_samples(images, labels, n=5):
    plt.figure(figsize=(8,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i][0]])
        plt.axis("off")
    plt.show()

show_samples(x_train, y_train)

# -----------------------------
# Step 3: Preprocess Data
# -----------------------------
# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# -----------------------------
# Step 4: Build CNN Model
# -----------------------------
def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_cnn()
model.summary()

# -----------------------------
# Step 5: Compile & Train Model
# -----------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# -----------------------------
# Step 7: Plot Accuracy & Loss
# -----------------------------
def plot_history(hist):
    plt.figure(figsize=(12,5))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

    plt.show()

plot_history(history)

# -----------------------------
# Step 8: Prediction Function
# -----------------------------
def predict_image(index):
    img = x_test[index]
    pred_class = np.argmax(model.predict(img.reshape(1,32,32,3)))
    true_class = np.argmax(y_test[index])

    plt.imshow(img)
    plt.title(f"Predicted: {class_names[pred_class]}\nActual: {class_names[true_class]}")
    plt.axis("off")
    plt.show()

# Example predictions
predict_image(23)
predict_image(34)
