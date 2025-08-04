# inference.py
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Import the model architecture from your model.py file
from model import ConditionalUNet

def get_device():
    """Returns the device to be used for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, config, device):
    """Loads the trained model from a checkpoint."""
    model = ConditionalUNet(
        n_channels=1,
        n_classes=3,
        num_colors=len(config['colour_names']),
        embed_dim=config['embed_dim']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, img_size):
    """Loads and preprocesses a single input image."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def infer(model, input_tensor, color_idx, device):
    """Performs inference on a single image."""
    input_tensor = input_tensor.to(device)
    color_idx_tensor = torch.tensor([color_idx], device=device)

    with torch.no_grad():
        prediction = model(input_tensor, color_idx_tensor)

    # Post-process the output
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = np.transpose(prediction, (1, 2, 0)) # C, H, W -> H, W, C
    prediction = np.clip(prediction, 0, 1) # Ensure values are in a valid range
    return prediction

def visualize_prediction(input_img_path, prediction, colour_name):
    """Visualizes the input and the predicted output."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Load and display the original input image
    input_img = Image.open(input_img_path)
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f"Input ({colour_name})")
    axes[0].axis('off')
    
    # Display the predicted output
    axes[1].imshow(prediction)
    axes[1].set_title(f"Prediction for {colour_name}")
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    # Define the same config used for training
    config = {
        "img_size": 64,
        "embed_dim": 256,
        # You'll need the list of colour names from your training script or dataset
        "colour_names": ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta", "lime", "pink"]
    }

    # Define paths
    model_path = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\wandb\models\conditional_unet_50mod.pth'

    # Example input data
    input_image_path = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\dataset\validation\inputs\octagon.png' # Replace with the path to a grayscale polygon image
    desired_colour_name = 'red' # Choose a color from the list

    # Get device
    device = get_device()

    # Load model
    model = load_model(model_path, config, device)

    # Preprocess input image
    input_tensor = preprocess_image(input_image_path, config['img_size'])
    
    # Get the color index
    colour_map = {name: i for i, name in enumerate(config['colour_names'])}
    color_idx = colour_map[desired_colour_name]

    # Run inference
    prediction = infer(model, input_tensor, color_idx, device)

    # Visualize the result
    visualize_prediction(input_image_path, prediction, desired_colour_name)