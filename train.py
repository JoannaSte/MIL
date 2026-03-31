import argparse
import lightning as pl
import os
import torch
from Learner import Learner
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from model import MILModel
from create_dataloader import MnistBags


def main():
    # Define arguments that can be passed to the script via the command line
    parser = argparse.ArgumentParser(description='Train a deep MIL model on selected numbers from MNIST dataset for number 0 prediction')
    parser.add_argument('--model_save_dir', default=None, help='Directory to save the trained model checkpoints.', dest='model_save_dir')
    parser.add_argument('--metrics_save_dir', default=None, help='Directory to save model training metrics.', dest='metrics_save_dir')
    parser.add_argument('--bag_length', default='100', type=int, help='The length of a bag (number of instances in a bag).', dest='bag_length')
    parser.add_argument('--num_features', default='32', type=int, help='Number of features extracted from the images.', dest='num_features')
    parser.add_argument('--num_bins', default='10', type=int, help='Number of bins in the Distribution Pooling filter.', dest='num_bins')
    parser.add_argument('--sigma', default='0.05', type=float, help='Sigma value in the Distribution Pooling filter.', dest='sigma')
    parser.add_argument('--batch_size', default='1', type=int, help='Batch size (number of bags processed at once).', dest='batch_size')
    parser.add_argument('--num_workers', default='10', type=int, help='Number of subprocesses to use for data loading.', dest='num_workers')
    parser.add_argument('--learning_rate', default='1e-4', type=float, help='Learning rate.', dest='learning_rate')
    parser.add_argument('--num_epochs', default='50', type=int, help='Number of training epochs.', dest='num_epochs')
    parser.add_argument('--save_every_epochs', default='1', type=int, help='Model saving interval', dest='save_every_epochs')
    parser.add_argument('--weight_decay', default='1e-4', type=float, help='L2 regularization weight decay.', dest='weight_decay')
    parser.add_argument('--save_top_k_models', default='1', type=int, help='Number of best models to save based on validation loss.', dest='save_top_k_models')
    parser.add_argument('--device', default='cuda', help='Specify the device to use for training.', dest='device')
    
    # Parse the arguments
    args = parser.parse_args()

    # Ensure model_save_dir exists, if not, set it to the current directory
    if not args.model_save_dir or not os.path.isdir(args.model_save_dir):
        current_path = Path.cwd()
        args.model_save_dir = os.path.join(current_path, "saved_models")
    
    # Ensure metrics_save_dir exists, if not, set it to the current directory
    if not args.metrics_save_dir or not os.path.isdir(args.metrics_save_dir):
        current_path = Path.cwd()
        args.metrics_save_dir = os.path.join(current_path, "metrics")
      
    #Checking for GPU availability when selected for training; if no GPU is detected, the CPU will be used instead. 
    if args.device == 'cuda':
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
            args.num_workers = 0
    else:
        args.num_workers = 0

    # Print statements to confirm the creation of model and dataloaders
    print("Create model")
    model = MILModel(
                    num_features=args.num_features, 
                    num_bins=args.num_bins, 
                    sigma=args.sigma
                    )
    
    print("Create train dataloader")
    train_dataloader = DataLoader(
                                    MnistBags(
                                        train=True,
                                        bag_length=args.bag_length,
                                    ),
                                    batch_size = args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle = True
                                    )

    print("Create validation dataloader")
    validation_dataloader = DataLoader(
                                    MnistBags(
                                        train=False,
                                        bag_length=10,
                                    ),
                                    batch_size = args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle = True
                                    )
    
    # Initialize the logger for TensorBoard
    logger = TensorBoardLogger(save_dir=args.metrics_save_dir, name='logs')
    
    # Setup the checkpoint callback to save the model at specified intervals
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_save_dir,
        filename="model-{epoch:02d}-{validation_loss:.3f}",
        every_n_epochs=args.save_every_epochs,
        monitor="validation_loss",
        save_top_k=args.save_top_k_models,
    )
    
    # Setup EarlyStopping callback to stop training when validation loss stops improving
    early_stopping_callback = EarlyStopping(
        monitor="validation_loss",  
        patience=15,  # Number of epochs with no improvement after which training will stop
        mode="min",  # Mode 'min' means training will stop when validation loss stops decreasing
        verbose=True
    )

    # Initialize the Learner class, which handles the training process
    autoencoder = Learner(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Setup the trainer for PyTorch Lightning, specifying the number of epochs and logging
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],#, early_stopping_callback],
        max_epochs=args.num_epochs, 
        default_root_dir=args.model_save_dir, 
        log_every_n_steps=1,  # Log metrics every step,
        accelerator=args.device
    )
    
    print("Start Training")
    trainer.fit(model=autoencoder, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    print("End Training")

if __name__ == '__main__':
    main()