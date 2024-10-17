import matplotlib.pyplot as plt

# Accuracy values from the training log
history = {
    'accuracy': [0.6713, 0.7188, 0.7948, 0.8125, 0.8240, 0.7812, 0.8362, 0.8125, 0.8368, 0.8350],
    'val_accuracy': [0.6562, 0.8462, 0.7329, 0.9231, 0.6662, 0.6154, 0.6734, 0.7692, 0.7404, 0.8462]
}

# Number of epochs
epochs = range(1, 11)

# Plotting training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
