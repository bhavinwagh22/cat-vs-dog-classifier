import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions

# 1. Load the Base Model
# weights='imagenet' loads the pre-trained weights
# include_top=False removes the final 1,000-class classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Freeze the Base Model
# This prevents the ImageNet weights from being updated during the first training phase
base_model.trainable = False

# 3. Build the Custom Classifier on Top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),          # Reduces 7x7x2048 to a 1x2048 vector
    layers.Dense(256, activation='relu'),     # Extra learning layer
    layers.Dropout(0.5),                      # Prevents overfitting
    layers.Dense(10, activation='softmax')    # Final 10 classes
])

# 4. Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # Use this if labels are integers
    metrics=['accuracy']
)

model.summary()
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    preds = model.predict(img_preprocessed)
    class_index = np.argmax(preds[0])

    return class_index
