import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from timm import create_model


def objective(trial, num_classes, train_loader, val_loader, class_weights, device, num_epochs=10):
    """
    Objective function for Optuna hyperparameter search.

    Parameters
    ----------
    trial (optuna.trial.Trial): Optuna trial object
    num_classes (int): Number of classes in the dataset
    train_loader (torch.utils.data.DataLoader): The training data loader
    val_loader (torch.utils.data.DataLoader): The validation data loader
    class_weights (torch.Tensor): Class weights for weighted cross entropy loss
    device (torch.device): The device to run the model on
    num_epochs (int): Number of epochs to run the model for

    Returns
    ----------
    float: The macro F1 score of the model on the validation set
    """
    # 1. Hyperparameter search space
    model_name = trial.suggest_categorical('model_name', ['efficientnetv2_s', 'convnext_tiny'])
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    
    # 2. Build the model, optimizer, loss function, and learning rate scheduler
    model = create_model(model_name=model_name, pretrained=True, in_chans=1,
                         num_classes=num_classes).to(device)
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # weighted cross entropy loss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 3. The training loop (abbreviated)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 4. The validation loop
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # 5. Report result to optuna
        trial.report(macro_f1, epoch)
        
        # Handle pruning (stop unpromising trials early).
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # The final value to be optimized
    return macro_f1

# Running the experiment
if __name__ == "__main__":

    # Create a study object. target is to maximize the f1 score
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Start the optimization process. n different parameter set trials
    study.optimize(objective, n_trials=50)
    
    # Results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation Accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")