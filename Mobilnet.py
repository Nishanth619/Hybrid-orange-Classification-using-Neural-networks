import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set directories for training and validation datasets
train_dir =r"D:\Types of citus\CITRICOS_COL\train" # Replace with your training dataset path
val_dir = r"D:\Types of citus\CITRICOS_COL\test" # Replace with your validation dataset path

# Image dimensions for MobileNet
img_width, img_height = 224, 224  # MobileNet expects 224x224 input images

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=64,  # Adjust batch size as needed
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=64,  # Adjust batch size as needed
    class_mode='categorical'
)

# Load MobileNet model pre-trained on ImageNet and exclude the top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Add dropout to reduce overfitting
predictions = Dense(8, activation='softmax')(x)  # 8 classes

# Create a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base layers of the model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model in Keras format
model.save('hybrid_orange_mobilenet(90).keras')
