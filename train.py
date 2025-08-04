import os
import torch
import torch.nn as nn # Added nn import
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from model import ConditionalUNet
from dataset import get_dataloaders, visualize_dataset # Added visualize_dataset import

def train_model(config):
    # Initialize WandB run
    wandb.init(project="polygon-colorizer", config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data loaders
    train_loader, val_loader, colour_names = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"]
    )
    
    # Initialize model
    model = ConditionalUNet(
        n_channels=1, # Input is grayscale
        n_classes=3,  # Output is RGB
        num_colors=len(colour_names),
        embed_dim=config["embed_dim"]
    ).to(device)
    
    # Log model architecture to WandB
    wandb.watch(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss() # Or L1Loss
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    model_dir = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\wandb\models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "conditional_unet_50mod.pth")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for input_img, colour_idx, output_img in train_loader:
            input_img = input_img.to(device)
            colour_idx = colour_idx.to(device)
            output_img = output_img.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast():
                predictions = model(input_img, colour_idx)
                loss = criterion(predictions, output_img)
            
            # Scale the loss and call backward()
            scaler.scale(loss).backward()
            
            # Step with the optimizer
            scaler.step(optimizer)
            
            # Update the scaler for the next iteration
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop (also with autocast)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            with autocast():
                for input_img, colour_idx, output_img in val_loader:
                    input_img = input_img.to(device)
                    colour_idx = colour_idx.to(device)
                    output_img = output_img.to(device)
                    
                    predictions = model(input_img, colour_idx)
                    loss = criterion(predictions, output_img)
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics and example images to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "examples": [wandb.Image(input_img[0].float(), caption="Input"),
                         wandb.Image(output_img[0].float(), caption="Ground Truth"),
                         wandb.Image(predictions[0].float(), caption="Prediction")]
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
    
    print("Model saved successfully.")
    # Save the final model
    wandb.finish()

if __name__ == "__main__":
    # Hyperparameters for Google Colab
    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 4,  # SIGNIFICANTLY REDUCED BATCH SIZE
        "img_size": 64,  # REDUCED IMAGE SIZE
        "embed_dim": 256,  # Reduced embedding dimension
        "data_dir": r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\dataset'
    }
    
    # Initialize and run the model
    # It's a good idea to first check if CUDA is available and print device info
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    try:
        train_model(config)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nFATAL ERROR: Out of memory on GPU. Please try a smaller batch size or image size.")
            print("Current config:", config)
        else:
            raise e