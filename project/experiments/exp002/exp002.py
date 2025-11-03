"""
Experiment: exp-002 (H-02)

Hypothesis:
- 変更対象: モデル（バックボーンの初期化）
- 内容: 画像事前学習済み重みを有効化（pretrained=True）。
- 期待効果・理由: ImageNet 事前学習により特徴抽出の初期性能が向上し、
  学習初期の収束と汎化が改善して検証 MAE/MSE の低下が見込める。
- 評価指標: MAE（主）、MSE（副）。

注: 本実験では他のハイパーパラメータ・前処理は変更せず、
    単一変更の効果検証に限定します。
"""

# !pip install -q pytorch-lightning torchmetrics timm

import os
import glob
import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torchmetrics

print(f"PyTorch: {torch.__version__}")
print(f"Lightning: {pl.__version__}")
print(f"TIMM: {timm.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

## Explore Data

PATH_DATA = '/kaggle/input/csiro-biomass'
PATH_TRAIN_CSV = os.path.join(PATH_DATA, 'train.csv')
PATH_TRAIN_IMG = os.path.join(PATH_DATA, 'train')
PATH_TEST_IMG = os.path.join(PATH_DATA, 'test')

df = pd.read_csv(PATH_TRAIN_CSV)
print(f"Dataset size: {df.shape}")
df.head()

TARGET_COLS = [c for c in df.columns if c not in ['image_id', 'Image']]
print(f"Target columns: {TARGET_COLS}")
print(f"Number of targets: {len(TARGET_COLS)}")

# Exclude non-numeric or identifier columns from histogram plotting
cols_to_plot = [col for col in TARGET_COLS if col not in ['sample_id', 'image_path', 'State', 'target_name']]

for col in cols_to_plot:
    plt.figure(figsize=(8, 3)) # Create a new figure for each histogram
    plt.hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{col} Distribution', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right") # Rotate x-axis labels
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()
    
    
cols_to_plot = ['State', 'target_name']
n_rows, n_cols = 1, len(cols_to_plot)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

# Ensure axes is an array even for a single subplot
axes = axes.flatten()

for ax, col in zip(axes, cols_to_plot):
    counts = df[col].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.tight_layout()
plt.show()

# Convert 'Sampling_Date' to datetime objects
df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])

# Extract the day of the year
df['Day_of_Year'] = df['Sampling_Date'].dt.dayofyear

# Calculate the correlation between 'target' and 'Day_of_Year'
correlation = df['target'].corr(df['Day_of_Year'])

print(f"The correlation between 'target' and 'Day_of_Year' is: {correlation}")

def show_images(df_sample, n=12, path_img=PATH_DATA):
    """Displays a linear sampling of images sorted by target value."""

    # Sort the DataFrame by the 'target' column
    df_sorted = df_sample.sort_values(by='target').reset_index(drop=True)

    # Perform linear sampling
    indices_to_show = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
    df_to_show = df_sorted.iloc[indices_to_show]

    # Determine the number of rows and columns for subplots
    n_cols = 3  # You can adjust this number
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    # Remove unused subplots if any
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    for i, (idx, row) in enumerate(df_to_show.iterrows()):
        # Use image_path directly (includes train/ID....jpg)
        img_path = os.path.join(path_img, row['image_path'])

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            # Include the target value in the title
            title = f"ID: {row['sample_id']}\nTarget: {row['target']:.2f}"
            axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage: Show 12 images linearly sampled based on target value
show_images(df, n=12)

class BiomassDataset(Dataset):
    """Simple dataset for biomass regression."""

    def __init__(self, df, path_img, transforms=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.path_img = path_img
        self.transforms = transforms
        self.mode = mode
        # Assume target column exists for train mode, but not necessarily for test
        self.target_col = 'target' if mode == 'train' else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_relative_path = row['image_path']
        img_path = os.path.join(self.path_img, img_relative_path)
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if self.mode == 'test':
            # For test mode, return image and sample_id
            return img, img_relative_path

        # For train mode, return image and target
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        return img, target
    
    # Initialize the dataset (assuming you have a DataFrame 'df' and image path 'PATH_DATA')
# You might need to define transforms later when setting up the DataModule
dataset = BiomassDataset(df, PATH_DATA)

# Get three random indices
random_indices = random.sample(range(len(dataset)), 3)

# Display the random samples
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, idx in enumerate(random_indices):
    img, target = dataset[idx]
    # Convert the PyTorch tensor image back to PIL Image for displaying
    # This assumes the default tensor format from PILToTensor or similar
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy() # Assuming CxHxW format, convert to HxWxD

    axes[i].imshow(img)
    axes[i].set_title(f"Target: {target}") # Display targets
    # axes[i].axis('off')

plt.tight_layout()
plt.show()

from torchvision import transforms
from pathlib import Path


class BiomassDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, img_size=(456, 456), val_split=0.2):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None  # Add test_df
        # ImageNet standards
        self._color_mean = [0.485, 0.456, 0.406]
        self._color_std = [0.229, 0.224, 0.225]
        # Define the transforms
        self.transforms = transforms.Compose([
            transforms.Resize(img_size[0] * 2,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              max_size=int(img_size[0] * 2.5)), # Scale to max 1000x1000
            transforms.RandomResizedCrop(self.img_size), # Add random resized crop
            transforms.RandomHorizontalFlip(), # Add random horizontal flip
            transforms.RandomVerticalFlip(), # Add random vertical flip
            #transforms.RandomRotation(degrees=15), # Add random rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Add random color jitter with more parameters
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Add random affine transformations
            transforms.GaussianBlur(kernel_size=3), # Add Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize(mean=self._color_mean, std=self._color_std),
            # Note: Adding noise directly in transforms.Compose can be tricky with torchvision
            # For simple noise, you might add it as a custom transform or after ToTensor
            # Example of adding simple Gaussian noise after ToTensor:
            # lambda x: x + torch.randn_like(x) * 0.01 # Add Gaussian noise with small std
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(img_size[0] * 2,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              max_size=int(img_size[0] * 2.5)), # Scale to max 1000x1000
            transforms.CenterCrop(self.img_size), # Add random resized crop
            transforms.ToTensor(),
            transforms.Normalize(mean=self._color_mean, std=self._color_std),
        ])
        # Automatically determine image path and target columns
        self.df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        self._num_workers = os.cpu_count() if os.cpu_count() is not None else 0


    def setup(self, stage: Optional[str] = None):
        # Shuffle and split the DataFrame manually
        if stage == 'fit' or stage is None:
            shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            val_size = int(len(shuffled_df) * self.val_split)
            self.train_df = shuffled_df[:-val_size]
            self.val_df = shuffled_df[-val_size:]

        if stage == 'test' or stage is None:
            # For test data, we need to create a DataFrame from the image file paths
            test_image_dir = os.path.join(self.data_path, 'test')
            assert os.path.isdir(test_image_dir)
            test_image_paths = glob.glob(os.path.join(test_image_dir, '*.jpg'))
            # Extract sample_id from the image paths (assuming filename is sample_id.jpg)
            test_data = [{'sample_id': Path(p).stem, 'image_path': os.path.relpath(p, self.data_path)} for p in test_image_paths]
            self.test_df = pd.DataFrame(test_data)

    def train_dataloader(self):
        train_dataset = BiomassDataset(self.train_df, self.data_path, transforms=self.transforms, mode='train')
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self._num_workers)

    def val_dataloader(self):
        val_dataset = BiomassDataset(self.val_df, self.data_path, transforms=self.test_transforms, mode='train') # Use test_transforms for validation
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self):
        if self.test_df is None:
            self.setup(stage='test') # Ensure test_df is loaded

        test_dataset = BiomassDataset(self.test_df, self.data_path, transforms=self.test_transforms, mode='test')
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) # Set num_workers to 0 for testing
    
# Example Usage (assuming df, PATH_DATA and TARGET_COLS are defined)
data_module = BiomassDataModule(PATH_DATA, batch_size=8, img_size=(528, 528))
data_module.setup()

# You can now access the dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Example of getting a batch (optional)
train_images, train_targets = next(iter(train_loader))
print(f"Shape of training images batch: {train_images.shape}")
print(f"Shape of training targets batch: {train_targets.shape}")

# Get the first batch from the training dataloader
train_images, train_targets = next(iter(train_loader))

# Determine how many images to show (e.g., the first 4 from the batch)
n_images_to_show = min(4, train_images.shape[0])

fig, axes = plt.subplots(1, n_images_to_show, figsize=(4 * n_images_to_show, 5))

# Ensure axes is an array even for a single image
if n_images_to_show == 1:
    axes = [axes]

for i in range(n_images_to_show):
    img = train_images[i].permute(1, 2, 0).numpy() # Convert from CxHxW to HxWxD for displaying
    # Denormalize the image for better visualization (using ImageNet standards)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1) # Clip values to be between 0 and 1

    axes[i].imshow(img)
    # Assuming targets are a list of values for each image
    axes[i].set_title(f"Target: {train_targets[i].tolist()}")
    #axes[i].axis('off')

plt.tight_layout()
plt.show()

import pytorch_lightning as pl
import torch.nn as nn
import torch
import timm
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import the scheduler


class BiomassRegressionModel(pl.LightningModule):
    def __init__(self, model_name="tf_efficientnetv2_m", pretrained=True, num_targets=1, learning_rate=0.001): # Adjusted learning rate
        super().__init__()
        self.save_hyperparameters()

        # Load a pre-trained transformer model from timm
        # num_classes=0 removes the original classifier head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool='avg')

        # Add a regression head
        in_features = self.backbone.num_features
        self.regression_head = nn.Linear(in_features, num_targets)

        # Loss function
        self.criterion = nn.SmoothL1Loss()

        # Metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        # Pass the input through the backbone
        features = self.backbone(x)
        # Pass the features through the regression head
        output = self.regression_head(features)
        return output

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), targets.squeeze()) # Squeeze to match shapes if necessary
        self.train_mae(outputs.squeeze(), targets.squeeze())
        self.train_mse(outputs.squeeze(), targets.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', self.train_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', self.train_mse, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), targets.squeeze())
        self.val_mae(outputs.squeeze(), targets.squeeze())
        self.val_mse(outputs.squeeze(), targets.squeeze())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mae', self.val_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.val_mse, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for the test set."""
        images, sample_path = batch
        outputs = self(images)
        # Return predictions and image path
        return outputs.squeeze(), sample_path


    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Configure the learning rate scheduler
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'interval': 'epoch', # Adjust learning rate every epoch
            'frequency': 1,
            'monitor': 'val_loss_epoch' # Monitor validation loss for learning rate reduction
        }
        return [optimizer], [scheduler]


# Example of initializing the model
model = BiomassRegressionModel(model_name="tf_efficientnet_b6", pretrained=True)
# print(model)

from pytorch_lightning.loggers import CSVLogger

# Initialize the CSVLogger
logger = CSVLogger("logs", name="biomass_regression")

# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=20, # You can adjust the number of epochs
    logger=logger,
    accelerator='auto', # Use auto to automatically select accelerator (GPU/CPU)
    devices='auto', # Use auto to automatically select devices
    precision='16-mixed', # Use Automatic Mixed Precision (AMP)
    log_every_n_steps=5 # Update progress bar every 5 steps
)

# Fit the model
trainer.fit(model, data_module)

# Define the path to save the model
model_save_path = "biomass_regression_model.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Read the metrics.csv using the trainer's logger directory
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

# Remove the step column and set epoch as index
metrics.set_index("epoch", inplace=True)

display(metrics.dropna(axis=1, how="all").head())

# Melt the DataFrame to long-form for plotting
metrics_melted = metrics.reset_index().melt(id_vars='epoch', var_name='metric', value_name='value')

# Define metric groups
metric_groups = {
    'Loss': ['train_loss_step', 'val_loss_epoch'],
    'MAE': ['train_mae_step', 'val_mae_epoch'],
    'MSE': ['train_mse_step', 'val_mse_epoch']
}

# Plot metrics for each group in a separate chart
for title, metric_list in metric_groups.items():
    # Filter melted DataFrame for the current group
    group_metrics = metrics_melted[metrics_melted['metric'].isin(metric_list)]

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=group_metrics, x='epoch', y='value', hue='metric')
    plt.title(f'{title} over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.show()
    
    
# Assuming data_module is already initialized and setup has been called for the test stage
# If not, uncomment and run the DataModule initialization and setup cells first
# data_module = BiomassDataModule(PATH_DATA, batch_size=64)
# data_module.setup(stage='test')

# Get the test dataloader
test_loader = data_module.test_dataloader()

# Get the first batch from the test dataloader
test_images, test_sample_ids = next(iter(test_loader))

# Determine how many images to show (e.g., the first 4 from the batch)
n_images_to_show = min(4, test_images.shape[0])

fig, axes = plt.subplots(1, n_images_to_show, figsize=(4 * n_images_to_show, 5))

# Ensure axes is an array even for a single image
if n_images_to_show == 1:
    axes = [axes]

for i in range(n_images_to_show):
    img = test_images[i].permute(1, 2, 0).numpy() # Convert from CxHxW to HxWxD for displaying
    # Denormalize the image for better visualization (using ImageNet standards)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1) # Clip values to be between 0 and 1

    axes[i].imshow(img)
    # Display the sample_id for test images
    axes[i].set_title(test_sample_ids[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Assuming 'trainer' and 'model' objects are available from previous cells

# Need to re-initialize the test dataloader to get the sample_ids in order
test_loader = data_module.test_dataloader()

# Debugging: Print the type and content of test_loader
print(f"Type of test_loader: {type(test_loader)}")
print(f"Content of test_loader: {test_loader}")


# Generate predictions on the test set by manually iterating through the dataloader
all_predictions = []
all_image_paths = []

model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculation
    for images, img_path in test_loader:
        # Move images to the same device as the model
        images = images.to(model.device)
        outputs = model(images)
        # Convert outputs to a list of values
        outputs = outputs.squeeze().tolist()
        if not isinstance(outputs, list):
            # edge case with just single image
            outputs = [outputs]
        all_predictions.extend(outputs) # Flatten and convert to list
        all_image_paths.extend(img_path)


# Create a submission DataFrame
prediction_df = pd.DataFrame({'image_path': all_image_paths, 'target': all_predictions})

# Display the first few rows of the submisshttps://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/69009d67bee1b2807736006e_0637_One_Republic_September_2025_METTY_%20copy.jpgion DataFrame
print("Submission DataFrame head:")
display(prediction_df.head())

# You can save the submission_df to a CSV file in the required format
# submission_df.to_csv('submission.csv', index=False)

# prevent any negative values
prediction_df[prediction_df["target"] < 0]["target"] = 0

# Define the path to the sample submission file
test_csv_path = os.path.join(PATH_DATA, 'test.csv')

# Load the sample submission file
test_csv = pd.read_csv(test_csv_path)
# display(test_csv.head())

# del sample_submission_df['target']
test_csv = test_csv.merge(prediction_df, on='image_path', how='left')
display(test_csv.head())

# dump prediction into CSV file
test_csv[["sample_id", "target"]].to_csv('submission.csv', index=False)

! head submission.csv
