import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import seaborn as sns

def plot_training_history(history_path, output_path='training_plot.png'):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # If you have accuracy in the history, you can plot it as well
    if 'train_accuracy' in history and 'val_accuracy' in history:
        train_accuracy = history['train_accuracy']
        val_accuracy = history['val_accuracy']
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, label='Training Accuracy')
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def overall_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return np.mean(class_accuracies)

def kappa_coefficient(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.show()