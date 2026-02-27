import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1️⃣ Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2️⃣ Reshape (CNN ko 4D input chahiye)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 3️⃣ Normalize (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4️⃣ One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 5️⃣ Build CNN Model
model = Sequential([
    
    # First Convolution Layer
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    
    # Second Convolution Layer
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    # Flatten Layer
    Flatten(),
    
    # Fully Connected Layer
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Output Layer
    Dense(10, activation='softmax')
])

# 6️⃣ Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7️⃣ Train Model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# 8️⃣ Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# 9️⃣ Plot Loss
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()