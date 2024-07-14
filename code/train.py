import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_model(model, train_loader, val_loader, num_epochs, lr, optim_name, clip, when,patience=10,model_path='../model/best_model.pth', history_path='../history/training_history.json', resume_training=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optim_name, model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=when, gamma=0.1)

    best_val_loss = float('inf')

    start_epoch = 0
    best_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    if resume_training and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss}")

    patience_counter = 0

    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training: Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        scheduler.step()

        val_loss,val_accuracy = evaluate_model(model, val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2f}%')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with validation loss: {best_val_loss}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1} with validation loss: {best_val_loss}')
                break        
        
        if epoch%100 == 0 and epoch!=0 :
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss
            }
            check_name = model_path.rstrip('.pth')
            torch.save(checkpoint, f'{check_name}_checkpoint_{epoch}.pth')
            


        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

    
    # Save the training history to a file
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # Return the path to the best model and training history
    return model_path, history_path


def evaluate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    val_loss /= len(val_loader)

    return val_loss,accuracy
