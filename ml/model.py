"""The class notebook's two-convolution CNN, with training-only regularization."""
import tensorflow as tf

def build_model(seed=42):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(64, 64, 3)),
        tf.keras.layers.RandomFlip('horizontal', seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed + 1),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3, seed=seed + 2),
        tf.keras.layers.Dense(1, activation='sigmoid', name='probability_without_mask'),
    ], name='vision03_mask_cnn')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
