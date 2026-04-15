import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt
import os
from src.model import NextWordLSTM
from src.utils import calculate_perplexity

with open("config.yaml") as f:
    config = yaml.safe_load(f)

def train():
    wandb.init(project=config["training"]["wandb_project"],
               name=config["training"]["wandb_run_name"],
               config=config)
    
    # Load data
    train_x, train_y, val_x, val_y = torch.load("data/processed/data_tensors.pt")
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordLSTM(config["data"]["vocab_size"], 
                         config["model"]["embed_dim"],
                         config["model"]["hidden_dim"],
                         config["model"]["num_layers"],
                         config["model"]["dropout"]).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output, _ = model(batch_x)
                val_loss += criterion(output, batch_y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        perplexity = calculate_perplexity(val_loss)
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": perplexity,
            "epoch": epoch + 1
        })
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print("Early stopping!")
                break
    
    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("logs/loss_curve.png")
    wandb.log({"loss_curve": wandb.Image("logs/loss_curve.png")})
    plt.close()

    wandb.finish()
    print("Training complete! Best model saved.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train()