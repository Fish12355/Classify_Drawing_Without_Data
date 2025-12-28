import tensorflow as tf
from tensorflow.keras import layers, models

old_model = tf.keras.models.load_model("quickdraw_model.keras", compile=False)

new_model = models.Sequential()
new_model.add(layers.Input(shape=(28,28,1)))

for layer in old_model.layers:
    if layer.__class__.__name__ == "Sequential":
        continue
    else:
        new_model.add(layer)

new_model.save("quickdraw_model_clean.h5")

