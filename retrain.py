import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from model import ConditionalUNet
from dataset import get_dataloaders, visualize_dataset

def transfer_learn_model(config, pretrained_model_path):
    # Initialize WandB run
    wandb.init(project="polygon-colorizer-transfer-learning", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    train_loader, val_loader, colour_names = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"]
    )

    # Initialize model
    model = ConditionalUNet(
        n_channels=1,
        n_classes=3,
        num_colors=len(colour_names),
        embed_dim=config["embed_dim"]
    ).to(device)

    # --- Transfer Learning Step ---
    print(f"Loading pre-trained model from {pretrained_model_path}...")
    try:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print("Pre-trained weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise e

    # Log model architecture to WandB
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # The rest of the training loop for fine-tuning
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for input_img, colour_idx, output_img in train_loader:
            input_img = input_img.to(device)
            colour_idx = colour_idx.to(device)
            output_img = output_img.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                predictions = model(input_img, colour_idx)
                loss = criterion(predictions, output_img)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
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
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "examples": [wandb.Image(input_img[0].float(), caption="Input"),
                         wandb.Image(output_img[0].float(), caption="Ground Truth"),
                         wandb.Image(predictions[0].float(), caption="Prediction")]
        })
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth"))
    wandb.finish()


if __name__ == "__main__":
    # Hyperparameters
    config = {
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 4,
        "img_size": 64,
        "embed_dim": 128,
        "data_dir": r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\dataset'
    }
    
    # Set this path to the .pth file you want to use for transfer learning
    # Example:
    # pretrained_model_path = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\wandb\run-20231027_103045-xyz\files\model.pth'
    pretrained_model_path = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\Ayna UNET Model\wandb\models\conditional_unet_50epochs.pth'

    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    try:
        transfer_learn_model(config, pretrained_model_path)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nFATAL ERROR: Out of memory on GPU. Please try a smaller batch size or image size.")
            print("Current config:", config)
        else:
            raise e