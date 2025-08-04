# filename: dataset.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

class PolygonDataset(Dataset):
    def __init__(self, data_dir, split, img_size=256):
        self.data_dir = os.path.join(data_dir, split)
        self.input_dir = os.path.join(self.data_dir, "inputs")
        self.output_dir = os.path.join(self.data_dir, "outputs")
        
        with open(os.path.join(self.data_dir, "data.json"), 'r') as f:
            self.data = json.load(f)
        
        self.colour_map = {
            "red": 0, "green": 1, "blue": 2, "yellow": 3, "orange": 4, 
            "purple": 5, "cyan": 6, "magenta": 7, "lime": 8, "pink": 9
        }
        self.colour_names = list(self.colour_map.keys())
        
        self.transform_input = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.transform_output = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # CORRECTED LOOP
        self.items = []
        # Check if the data is a dictionary or a list
        if isinstance(self.data, list):
            for item_dict in self.data:
                self.items.append({
                    "input_polygon": item_dict["input_polygon"],
                    "colour": item_dict["colour"],
                    "output_image": item_dict["output_image"]
                })
        # If it's a dictionary (for some reason, old code)
        elif isinstance(self.data, dict):
            for input_polygon, info in self.data.items():
                self.items.append({
                    "input_polygon": input_polygon,
                    "colour": info["colour"],
                    "output_image": info["output_image"]
                })
        else:
            raise TypeError("data.json file is not a list or dictionary.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load images
        input_path = os.path.join(self.input_dir, item["input_polygon"])
        output_path = os.path.join(self.output_dir, item["output_image"])
        
        input_polygon = Image.open(input_path).convert("RGB")
        output_image = Image.open(output_path).convert("RGB")
        
        # Apply transformations
        input_polygon_tensor = self.transform_input(input_polygon)
        output_image_tensor = self.transform_output(output_image)
        
        # Get colour index
        colour_name = item["colour"]
        colour_index = torch.tensor(self.colour_map[colour_name], dtype=torch.long)
        
        return input_polygon_tensor, colour_index, output_image_tensor

def get_dataloaders(data_dir, batch_size, img_size):
    train_dataset = PolygonDataset(data_dir, "training", img_size=img_size)
    val_dataset = PolygonDataset(data_dir, "validation", img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for Colab
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers=0 for Colab
    
    return train_loader, val_loader, train_dataset.colour_names

# The visualization function now takes colour_names as an argument
def visualize_dataset(dataloader, colour_names, num_samples=4):
    """
    Visualizes a batch of data from the DataLoader.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader to visualize.
        colour_names (list): List of colour names corresponding to indices.
        num_samples (int): The number of samples to display.
    """
    # Get a batch of data
    input_polygons, colour_indices, output_images = next(iter(dataloader))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    fig.suptitle("Dataset Visualization", fontsize=16)

    for i in range(num_samples):
        # Convert tensors to numpy arrays and adjust dimensions for plotting
        input_polygon = input_polygons[i].squeeze().numpy()
        output_image = np.transpose(output_images[i].numpy(), (1, 2, 0))
        colour_name = colour_names[colour_indices[i].item()]
        
        # Plot the input image
        axes[i, 0].imshow(input_polygon, cmap='gray')
        axes[i, 0].set_title(f"Input ({colour_name})")
        axes[i, 0].axis('off')
        
        # Plot the output image
        axes[i, 1].imshow(output_image)
        axes[i, 1].set_title("Output (Ground Truth)")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "./dataset" # Use this for local execution
    # For Google Colab, you might need to use: DATA_DIR = "/content/dataset"

    train_loader, _, colour_names = get_dataloaders(data_dir=DATA_DIR, batch_size=8, img_size=256)
    visualize_dataset(train_loader, colour_names)