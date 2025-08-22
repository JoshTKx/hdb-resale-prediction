import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
  device = next(model.parameters()).device  # Get model's device
  save_dir = '/content/drive/My Drive/HDB_resale_project/model_checkpoints' # Define save_dir
  os.makedirs(save_dir, exist_ok=True)

  # Initialize optimizer and loss function
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization
  criterion = nn.MSELoss()

  # Learning rate scheduler (reduce when validation plateaus)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=patience//2
  )

  # Training tracking
  train_losses = []
  val_losses = []
  val_rmses = []
  best_val_loss = float('inf')
  patience_counter = 0
  best_model_state = None
  best_metrics = {} # Initialize best_metrics

  print(f"\nStarting training on {device}")
  print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

  for epoch in range(epochs):
    # Training phase
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

    # Validation phase
    val_loss, val_predictions, val_targets = validate(model, val_loader, criterion, device)

    # Calculate validation RMSE
    val_rmse = np.sqrt(val_loss)  # Since we're using MSE loss

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Track metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_rmses.append(val_rmse)

    # Early stopping logic
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      best_model_state = model.state_dict().copy()  # Save best model

      best_metrics = {
              'val_loss': val_loss,
              'val_rmse': val_rmse,
              'epoch': epoch + 1
      }

      model.save_model(
          f'{save_dir}/best_hdb_model.pth',
          optimizer,
          epoch + 1,
          best_metrics
      )
    else:
      patience_counter += 1

    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, Val RMSE: ${val_rmse:,.0f}, LR: {current_lr:.6f}")

    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        break

  # Load best model state
  if best_model_state is not None:
      model.load_state_dict(best_model_state)
      print("Loaded best model weights")
  else:
    print("No improvement during training - using final weights")
    best_val_loss = val_loss  # Update best_val_loss for consistency

  final_metrics = {
        'final_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'best_val_loss': best_val_loss
    }
  model.save_model(f'{save_dir}/final_hdb_model.pth', optimizer, epoch + 1, final_metrics)


  return {
      'train_losses': train_losses,
      'val_losses': val_losses,
      'val_rmses': val_rmses,
      'best_val_loss': best_val_loss
  }


def train_epoch(model, train_loader, optimizer, criterion, device):

  model.train()
  total_loss = 0
  num_batches = 0

  for cat_batch, cont_batch, target_batch in train_loader:
    # Move data to device (GPU/MPS/CPU)
    cat_batch = cat_batch.to(device)
    cont_batch = cont_batch.to(device)
    target_batch = target_batch.to(device)

    # Zero gradients from previous iteration
    optimizer.zero_grad()

    # Forward pass
    predictions = model(cat_batch, cont_batch)

    # Calculate loss
    loss = criterion(predictions, target_batch)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1

  return total_loss / num_batches


def validate(model, val_loader, criterion, device):
  model.eval()
  total_loss = 0
  all_predictions = []
  all_targets = []


  with torch.inference_mode():  # Don't calculate gradients during validation
    for cat_batch, cont_batch, target_batch in val_loader:
        cat_batch = cat_batch.to(device)
        cont_batch = cont_batch.to(device)
        target_batch = target_batch.to(device)

        predictions = model(cat_batch, cont_batch)
        loss = criterion(predictions, target_batch)

        total_loss += loss.item()
        all_predictions.append(predictions.cpu())
        all_targets.append(target_batch.cpu())

  # Concatenate all batches
  all_predictions = torch.cat(all_predictions)
  all_targets = torch.cat(all_targets)

  return total_loss / len(val_loader), all_predictions, all_targets