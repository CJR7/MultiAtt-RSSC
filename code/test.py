import torch
from torch.utils.data import DataLoader
from utils import overall_accuracy, average_accuracy, kappa_coefficient, plot_confusion_matrix
import json
import numpy as np

def test_model(model, test_loader, classes, confusion_matrix_path='confusion_matrix.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    overall_acc = overall_accuracy(all_labels, all_preds)
    avg_acc = average_accuracy(all_labels, all_preds)
    kappa = kappa_coefficient(all_labels, all_preds)
    
    # Print metrics
    #print(f'Overall Accuracy: {overall_acc}')
    #print(f'Average Accuracy: {avg_acc}')
    #print(f'Kappa Coefficient: {kappa}')
    
    # Plot and save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes, output_path=confusion_matrix_path)
    
    return overall_acc, avg_acc, kappa
