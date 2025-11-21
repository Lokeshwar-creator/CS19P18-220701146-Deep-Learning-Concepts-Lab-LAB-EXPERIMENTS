# -----------------------------------------------------------
# AIM: To build autoencoders with Keras/TensorFlow
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------
# 1. Download and load the dataset
# -----------------------------------------------------------
(x_train, _), (x_test, _) = mnist.load_data()

# -----------------------------------------------------------
# 2. Analysis & preprocessing of the dataset
# -----------------------------------------------------------
# Normalize (0–1)
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# Flatten (28×28 → 784)
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# -----------------------------------------------------------
# 3. Build a simple autoencoder model using Keras/TensorFlow
# -----------------------------------------------------------
input_dim = 784
encoding_dim = 64  # compressed feature size

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)

# Decoder
decoded = Dense(input_dim, activation="sigmoid")(encoded)

# Autoencoder
autoencoder = Model(input_layer, decoded)

# Encoder model separately (optional)
encoder = Model(input_layer, encoded)

# -----------------------------------------------------------
# 4. Compile & fit the model
# -----------------------------------------------------------
autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.2
)

# -----------------------------------------------------------
# 5. Perform prediction with the test dataset
# -----------------------------------------------------------
decoded_imgs = autoencoder.predict(x_test)

# -----------------------------------------------------------
# 6. Calculate performance metrics (MSE, SSIM)
# -----------------------------------------------------------
mse_scores = []
ssim_scores = []

for i in range(100):  # calculate for 100 samples
    orig = x_test[i].reshape(28, 28)
    recon = decoded_imgs[i].reshape(28, 28)

    mse_scores.append(mean_squared_error(orig, recon))
    ssim_scores.append(ssim(orig, recon, data_range=1.0))

print("Average MSE:", np.mean(mse_scores))
print("Average SSIM:", np.mean(ssim_scores))

# -----------------------------------------------------------
# Display original vs reconstructed images
# -----------------------------------------------------------
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()
