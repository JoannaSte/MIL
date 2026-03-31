# Applying the Multiple Instance Learning (MIL) approach to prediction tasks using the MNIST dataset

This work presents an example of using the MIL (Multiple Instance Learning) model for digit classification. The experiments were conducted using the MNIST dataset. 
In this study, instances of digits 0 and 7 were grouped into bags, and the objective was to predict the percentage of digit 0 present in each bag.

## MNIST Dataset

The MNIST dataset (Modified National Institute of Standards and Technology dataset) is one of the most widely used datasets in machine learning and computer vision. It consists of grayscale images of handwritten digits ranging from 0 to 9. In this research, we use only the digits 0 and 7 . Here’s a detailed overview of the used dataset:

Key Features:
1. Image Size: Each image is 28x28 pixels, with each pixel represented as a grayscale intensity value ranging from 0 (black) to 255 (white).
2. Number of Samples:
- Training set: 12,188 images (0: 5923, 7: 6265).
- Test set: 2,008 images (0: 980, 7: 1028).
3. Classes: 2 classes, corresponding to the digits 0 and 7.
4. Labels: Each image is labeled with the correct digit.

## The presented work includes the following files. 

| File                                | Description                                                                                              |
|-------------------------------------|----------------------------------------------------------------------------------------------------------|
| `--train.py`                        | Sets all training parameters, loads the model and data, and starts the training of the defined model.    |
| `--create_dataloader.py`            | Contains the class responsible for loading the data.                                                     |
| `--distribution_pooling_filter.py`  | Includes the class that applies the Distribution Pooling Filter.                                         |
| `--model.py`                        | Houses the entire model framework.                                                                       |
| `--visualize_results.py`            | Enables the visualization of the results                                                                 |
| `--requirements.txt`                | Lists the packages used, useful for recreating the code on a different device.                           |
| `--README.md`                       | Discusses the entire code and provides instructions on how to use it.                                    |

Additionally, a `test` folder is attached, which contains an example model and training logs.

## Required Python Packages

Experiments were conducted within a pip-managed virtual environment on a Linux system.

To install requirements:

```console
pip install -r requirements.txt
```

## Multiple Instance Model
Multiple Instance Learning (MIL) organizes data into bags of instances, with labels assigned to bags rather than individual instances. MIL predicts bag-level labels by aggregating information from instances and is useful in weakly supervised tasks where instance-level labeling is impractical. Models often use pooling or attention mechanisms to combine instance information effectively. In this work, we utilized a Distribution Pooling Filter, which constructs histograms of the image features. Later, we used neural networks to transform the histograms into the final result.

### Dataloader
In this work, a dataloader was developed to load selected samples (numbers 0 and 7) from the MNIST dataset. In line with the assumptions of the MIL method, the data is returned as bags, each containing a definable number of images. The dataset is generated during the training process, allowing parameters such as the number of images per frame to be specified in the command that initiates the training, as detailed in the subsection below.

### Training
To train a model:
```console
python train.py
```
#### Available Arguments for `train.py`

You can customize the training process and data loading by specifying various arguments when running the script. Below is a list of the available arguments:

| Argument                                  | Description                                       | Default Value  |
|-----------------------------|-----------------------------------------------------------------|----------------|
| `--model_save_dir`          | Directory to save the trained model checkpoints.                | `None`         |
| `--metrics_save_dir`        | Directory to save model training metrics.                       | `None`         |
| `--bag_length`              | The length of a bag (number of instances in a bag).             | `100`          |
| `--num_features`            | Number of features extracted from the images.                   | `32`           |
| `--num_bins`                | Number of bins in the Distribution Pooling filter.              | `5`            |
| `--sigma`                   | Sigma value in the Distribution Pooling filter.                 | `0.05`         |
| `--batch_size`              | Batch size (number of bags processed at once).                  | `1`            |
| `--num_workers`             | Number of subprocesses to use for data loading.                 | `10`           |
| `--learning_rate`           | Learning rate.                                                  | `1e-4`         |
| `--num_epochs`              | Number of training epochs.                                      | `200`          |
| `--save_every_epochs`       | Model saving interval (in epochs).                              | `1`            |
| `--weight_decay`            | L2 regularization weight decay.                                 | `1e-4`         |
| `--save_top_k_models`       | Number of best models to save based on validation loss.         | `3`            |
| `--device`                  | Specify the device to use for training.                         | `cuda`         |


### Metrics
All metrics have been logged using tensorboard, to view them please use the method below:
```console
python -m tensorboard.main --logdir="path_to_dir_with_logs"
```
The code also provides the ability to create graphs from already trained models. When testing the model, make sure that the arguments `num_features`, and `num_bins` for the selected checkpoint are the same as those set in Visualize_results.py. These values affect the dimensions of certain layers in the model, and in case of a mismatch, the model will fail to load.

To create plots:
```console
python Visualize_results.py --model_dir="path_to_model_checkpoint" 
```
####  Available Arguments for `Visualize_results.py`

| Argument                     | Description                                                   | Default Value     |
|------------------------------|---------------------------------------------------------------|-------------------|
| `--model_dir`                | Path to the model to use.                                     | `test/model/model-epoch=08-validation_loss=0.127.ckpt`|
| `--save_path`                | Path to save the image.                                       | `None`            |
| `--file_name`                | Name of the file for saving the generated plot images.        | `results.jpg`     |
| `--bag_length`               | The length of a bag (number of instances in a bag).           | `100`             |
| `--num_features`             | Number of features extracted from the images.                 | `32`              |
| `--num_bins`                 | Number of bins in the Distribution Pooling filter.            | `10`              |
| `--sigma`                    | Sigma value in the Distribution Pooling filter.               | `05`              |                      
| `--num_workers`              | Number of subprocesses to use for data loading.               | `10`              |
| `--device`                   | Specify the device to use for training.                       | `cuda`            |

## Related works
  
```
@article{DBLP:journals/corr/abs-1802-04712,
  author       = {Maximilian Ilse and
                  Jakub M. Tomczak and
                  Max Welling},
  title        = {Attention-based Deep Multiple Instance Learning},
  journal      = {CoRR},
  volume       = {abs/1802.04712},
  year         = {2018},
  url          = {http://arxiv.org/abs/1802.04712},
  eprinttype    = {arXiv},
  eprint       = {1802.04712},
  timestamp    = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1802-04712.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}