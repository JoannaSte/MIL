import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model import MILModel
from Learner import Learner
from create_dataloader import MnistBags


parser = argparse.ArgumentParser(description='Make a visualization from the selected model on the validation dataset.')
parser.add_argument('--model_dir', default='test/model/model-epoch=08-validation_loss=0.127.ckpt', help='Path to the model to use', dest='model_dir')
parser.add_argument('--save_path', default=None, help='Path to save image', dest='save_path')
parser.add_argument('--file_name', default="results.jpg", help='Name of the file for saving the generated plot images', dest='file_name')
parser.add_argument('--bag_length', default='100', type=int, help='The length of a bag (number of instances in a bag).', dest='bag_length')
parser.add_argument('--num_features', default='32', type=int, help='Number of features extracted from the images.', dest='num_features')
parser.add_argument('--num_bins', default='10', type=int, help='Number of bins in the Distribution Pooling filter.', dest='num_bins')
parser.add_argument('--sigma', default='0.05', type=float, help='Sigma value in the Distribution Pooling filter.', dest='sigma')
parser.add_argument('--num_workers', default='10', type=int, help='Number of subprocesses to use for data loading.', dest='num_workers')
parser.add_argument('--device', default='cuda', help='Specify the device to use for training.', dest='device')

args = parser.parse_args()

if not args.model_dir or not os.path.isfile(args.model_dir):
    raise ValueError("Model not selected or does not exist. Please choose a valid model.")

if not args.save_path or not os.path.isdir(args.save_path):
    current_path = Path.cwd()
    args.save_path = os.path.join(current_path, args.file_name)
    
else:
    args.save_path = os.path.join(args.save_path, args.file_name)

if args.device == 'cuda':
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        args.num_workers = 0
else:
    args.num_workers = 0

        
# Create model to evaluate
print("Create model to evaluate")
model = MILModel( 
    num_features=args.num_features, 
    num_bins=args.num_bins, 
    sigma=args.sigma
)

print("Load weights from checkpoints")
model = Learner.load_from_checkpoint(args.model_dir, model=model, map_location=args.device)

print("Create validation dataloader")
validation_dataloader = DataLoader(
    MnistBags(
        train=False,
        bag_length=args.bag_length,
    ),
    batch_size=1,
    num_workers=args.num_workers,
    shuffle=True)

print("Create predictions and results")
# Lists to store results
pred_labels = []
true_labels = []

# Make predictions
for _, (data,label) in enumerate(validation_dataloader):
    data=data.to(args.device)
    predicted_value = model(data)
    pred_labels.append(predicted_value.cpu().squeeze().float().detach().numpy())
    true_labels.append(label.cpu().squeeze().float().detach().numpy())

# Convert to numpy arrays
true_values = np.array(true_labels)
predicted_values = np.array(pred_labels)

# Calculate residuals (prediction errors)
residuals = true_values - predicted_values

# Calculate mean and standard deviation of residuals
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

print("Create plots")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1st Plot: True vs Predicted values
max_val = max(np.max(true_values), np.max(predicted_values))
margin = max_val * 0.1
axs[0].set_xlim(0 - margin/5, max_val + margin)  
axs[0].set_ylim(0 - margin/5, max_val + margin)  
axs[0].scatter(true_values, predicted_values, color='purple', alpha=0.6)
axs[0].plot([0 - margin/5, max_val + margin], [0 - margin/5, max_val + margin], 'k--', label='y=x')
axs[0].set_xlabel('True values')
axs[0].set_ylabel('Predicted values')
axs[0].set_title('True vs Predicted Values')
axs[0].grid(True)


# 2nd Plot: True vs Residuals
axs[1].scatter(true_values, residuals, label='Residuals', color='purple', alpha=0.6)
axs[1].axhline(mean_residual, color='black', linestyle='--', label='Mean')
axs[1].axhline(mean_residual + std_residual, color='gray', linestyle='--', label='Mean + std')
axs[1].axhline(mean_residual - std_residual, color='gray', linestyle='--', label='Mean - std')
axs[1].set_xlabel('True values')
axs[1].set_ylabel('Residuals')
axs[1].set_title('True values vs Residuals')
axs[1].grid(True)
axs[1].legend()

print("Save and show plot")
plt.tight_layout()
plt.savefig(args.save_path)
plt.show()