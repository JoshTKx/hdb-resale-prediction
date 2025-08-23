import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred, split_name=""):
    """Calculate comprehensive regression metrics."""
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    mape = np.mean(np.abs((y_true_np - y_pred_np) / np.maximum(y_true_np, 1))) * 100
    r2 = r2_score(y_true_np, y_pred_np)

    metrics = {
        'RMSE': rmse,
        'MAE': mae, 
        'MAPE': mape,
        'R²': r2
    }

    if split_name:
        print(f"\n{split_name} Metrics:")
        for metric, value in metrics.items():
            if metric in ['RMSE', 'MAE']:
                print(f"  {metric}: ${value:,.0f}")
            elif metric == 'MAPE':
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")

    return metrics

def plot_training_curves(history):
    """Plot training progress visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(len(history['train_losses']))

    # Loss curves
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # RMSE curve
    ax2.plot(epochs, [np.sqrt(loss) for loss in history['train_losses']], 'b-', label='Training RMSE')
    ax2.plot(epochs, history['val_rmses'], 'r-', label='Validation RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE ($)')
    ax2.set_title('RMSE Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, split_name=""):
    """Plot predicted vs actual prices."""
    plt.figure(figsize=(10, 8))
    
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

    plt.scatter(y_true_np, y_pred_np, alpha=0.5)
    plt.plot([y_true_np.min(), y_true_np.max()], [y_true_np.min(), y_true_np.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'{split_name} - Predicted vs Actual Prices')
    plt.grid(True)

    r2 = r2_score(y_true_np, y_pred_np)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def convert_remaining_lease_to_months(text):
    """Convert remaining lease text to numerical months."""
    text = text.split()
    months = 0
    for i, word in enumerate(text):
        if word in ["years", "year"]:
            months += int(text[i-1]) * 12
        elif word in ["months", "month"]:
            months += int(text[i-1])
    return months

def get_device():
  if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
  elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
  else:
    device = torch.device('cpu')
    print("Using CPU")
  return device