import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = tf.keras.models.load_model(r'C:\Users\ADMIN\PycharmProjects\Citrus\optimized_cnn_model(70-30).keras')

# Image dimensions
img_width, img_height = 299, 299  # InceptionV3 input size

# Set validation dataset directory
val_dir = r"F:\dataset70-30\test"

# Data preprocessing for validation
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep order for correct predictions
)

# Predict the labels for the validation set
val_predictions = model.predict(validation_generator)
val_predicted_classes = np.argmax(val_predictions, axis=1)

# Get true labels from validation generator
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, val_predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()  

# Classification report
print("Classification Report:")
print(classification_report(true_classes, val_predicted_classes, target_names=class_labels))
