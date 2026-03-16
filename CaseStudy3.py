# Now Again train with some improvement
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2



# -------------------------------
# 1. Data Augmentation   increases dataset diversity
# -------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

# -------------------------------
# 2. CNN Model
# -------------------------------

model = Sequential()

# Conv Block 1
model.add(Conv2D(32,(3,3),
                 activation='relu',
                 kernel_regularizer=l2(0.001),
                 input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

# Conv Block 2
model.add(Conv2D(64,(3,3),
                 activation='relu',
                 kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(2,2))

# Conv Block 3
model.add(Conv2D(128,(3,3),
                 activation='relu',
                 kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

# Dropout to reduce overfitting
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(train_generator.num_classes, activation='softmax'))


# -------------------------------
# 3. Compile Model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# 4. Early Stopping
# -------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# -------------------------------
# 5. Train Model
# -------------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)


loss, accuracy = model.evaluate(val_generator)
print("Validation Accuracy:", accuracy)


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


