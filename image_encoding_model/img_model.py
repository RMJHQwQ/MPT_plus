from tqdm import tqdm
from transformers import ViTModel
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.models import resnet18
from PIL import Image

class ConvNet(nn.Module):
    def __init__(self, num_classes,dim):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 32 * 32, dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ImgModel(nn.Module):
    def __init__(self, num_classes, dim):
        super(ImgModel, self).__init__()
        self.local_model_path = "facebook/vjepa2-vitl-fpc64-256"
        self.cnn_model = ConvNet(num_classes,dim)
        self.vit_model = ViTModel.from_pretrained(self.local_model_path)
        self.vit_model.training = True
        self.vit_model.config.num_labels = 3
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, self.vit_model.config.num_labels)
        self.fc = nn.Linear(2048, dim)
        # self.adjust = nn.Linear(1536, num_classes)

    def forward(self, x):
        cnn_features = self.cnn_model(x)
        vit_features = self.vit_model(x).last_hidden_state[:,0,:]
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        feature = self.fc(combined_features)
        return feature