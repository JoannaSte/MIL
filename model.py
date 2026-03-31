import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from distribution_pooling_filter import DistributionPoolingFilter

class MnistResNet(ResNet):
    def __init__(self, num_classes):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):

        batch_size, num_images, channels, height, width = x.shape
        x = x.view(batch_size * num_images, channels, height, width)
        x = super(MnistResNet, self).forward(x)
        x = torch.softmax(x, dim=-1)
        x = x.view(batch_size, num_images, -1)

        return x

class FeatureExtractor(nn.Module):

	def __init__(self, num_features=32):
		super(FeatureExtractor, self).__init__()

		self._model_conv = MnistResNet(num_classes=num_features)
		self.relu = nn.ReLU()

	def forward(self, x):
     
		out = self._model_conv(x)
		out = self.relu(out)
  
		return out

class RepresentationTransformation(nn.Module):
	def __init__(self, num_features=32, num_bins=11, num_classes=1):
		super(RepresentationTransformation, self).__init__()

		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(num_features * num_bins, 192),
		    nn.LayerNorm(192),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(192, 96),
			nn.LayerNorm(96),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(96, num_classes)
			)

	def forward(self, x):

		out = self.fc(x)

		return out

class MILModel(nn.Module):

	def __init__(self, num_features=32, num_bins=11, sigma=0.1):
		super(MILModel, self).__init__()


		self._num_features = num_features
		self._num_bins = num_bins
		self._sigma = sigma
		self._feature_extractor = FeatureExtractor(num_features=num_features)
		self._mil_pooling_filter = DistributionPoolingFilter(num_bins=num_bins, sigma=sigma)
		self._representation_transformation = RepresentationTransformation(num_features=num_features, num_bins=num_bins, num_classes=1)

	def forward(self, x):

		out = self._feature_extractor(x)
		out = self._mil_pooling_filter(out)
		out = torch.flatten(out, 1)
		out = self._representation_transformation(out)
  
		return out