import glob
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

dataset_path = "data/npy100"
all_files = glob.glob(os.path.join(dataset_path,'*.npy'))
all_files = sorted(all_files)[:100]

X = np.empty([0,28*28], dtype='float32')
y = np.empty([0], dtype='int32')

max_images_per_class = 5000

classes = []

for idx, file in enumerate(all_files):
    data = np.load(file)

    data = data[:max_images_per_class]

    labels = np.full(data.shape[0],idx)

    X = np.vstack([X,data])
    y = np.hstack([y,labels])

    class_name,_ = os.path.splitext(os.path.basename(file))
    classes.append(class_name)

X = X.astype('float32')/255.0
X = X.reshape(-1,28,28,1)


X, y = shuffle(X,y,random_state=42)

data_augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.05, 0.05),
    layers.GaussianNoise(0.05)
])

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,

    #Block 1
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.15),

    #Block 2
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    #Block 3
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.35),

    layers.Flatten(),

    layers.Dense(512, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val,y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

model.save("quickdraw_model.keras")
