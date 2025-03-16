import os
import pandas as pd

# Function to create DataFrame from image directory
def create_df(base):
    dd = {"images": [], "labels": []}
    for label in os.listdir(base):
        label_path = os.path.join(base, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            dd["images"].append(img_path)
            dd["labels"].append(label)
    return pd.DataFrame(dd)

# Function to visualize results
def plot_results(train_loss, val_loss, train_acc, val_acc):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
