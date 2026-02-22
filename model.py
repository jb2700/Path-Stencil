import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
# import glob
import os
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class DenoiseUNet(nn.Module):
    def __init__(self):
        super(DenoiseUNet, self).__init__()
        self.enc1 = nn.Conv2d(7, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bot  = nn.Conv2d(256, 512, 3, padding=1)
        
        self.dec3 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.final = nn.Conv2d(64, 3, 1)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.relu(self.enc1(x))
        x2 = self.relu(self.enc2(nn.functional.max_pool2d(x1, 2)))
        x3 = self.relu(self.enc3(nn.functional.max_pool2d(x2, 2)))
        
        b  = self.relu(self.bot(nn.functional.max_pool2d(x3, 2)))
        
        up3 = nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.relu(self.dec3(torch.cat([up3, x3], dim=1)))
        up2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.relu(self.dec2(torch.cat([up2, x2], dim=1)))
        up1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.relu(self.dec1(torch.cat([up1, x1], dim=1)))
        return self.final(d1)

class PatchDataset(Dataset):
    def __init__(self, train_folder="train_data", test_folder="test_data", img_size=512, patch_size=256):
        self.img_size = img_size
        self.patch_size = patch_size
        self.train_folder = train_folder
        self.test_folder = test_folder

        self.allowed_ids = ['02', '03', '04', '06', '07']
        
        self.valid_ids = []
        for s_id in self.allowed_ids:

            noisy_path = os.path.join(self.train_folder, f"scene{s_id}_noisy.bin")
            gt_path = os.path.join(self.test_folder, f"test{s_id}_gt.bin")
            
            if os.path.exists(noisy_path) and os.path.exists(gt_path):
                self.valid_ids.append(s_id)
            else:
                print(f"Warning: Missing files for scene {s_id}. Skipping.")

        if not self.valid_ids:
            raise ValueError("No matching train/test pairs found! Check your folder paths and filenames.")
            
        print(f"Dataset initialized with {len(self.valid_ids)} scenes: {self.valid_ids}")

    def load_bin(self, path, channels):
        data = np.fromfile(path, dtype=np.float32)
        tensor = torch.from_numpy(data).view(self.img_size, self.img_size, channels).permute(2, 0, 1)

        return tensor.clamp(0, 1)

    def __len__(self):

        return 200 

    def __getitem__(self, idx):

        s_id = self.valid_ids[np.random.randint(0, len(self.valid_ids))]
        
        train_base = os.path.join(self.train_folder, f"scene{s_id}")
        noisy = self.load_bin(f"{train_base}_noisy.bin", 3)
        norm  = self.load_bin(f"{train_base}_normal.bin", 3)
        depth = self.load_bin(f"{train_base}_depth.bin", 1)
        
        gt_path = os.path.join(self.test_folder, f"test{s_id}_gt.bin")
        gt = self.load_bin(gt_path, 3)

        full_input = torch.cat([noisy, norm, depth], dim=0)

        h, w = self.img_size, self.img_size
        th, tw = self.patch_size, self.patch_size
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        
        return full_input[:, i:i+th, j:j+tw], gt[:, i:i+th, j:j+tw]

def train():
    os.makedirs("models", exist_ok=True)
    dataset = PatchDataset("train_data", "test_data")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = DenoiseUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    

    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    epochs = 50 
    print("Beginning Training (100 Epochs)...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            

            loss = (0.8 * criterion_l1(outputs, targets)) + (0.2 * criterion_mse(outputs, targets))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "models/denoiser_v2_deep.pth")
    print("Training Complete.")

def test_one_scene(scene_id, train_folder="train_data"):

    model = DenoiseUNet().to(device)
    # model.load_state_dict(torch.load("models/denoiser_v1.pth"))
    model.load_state_dict(torch.load("models/denoiser_v2_deep.pth"))
    model.eval()

 
    def load_full_normalized(path, c):
        data = np.fromfile(path, dtype=np.float32)
        tensor = torch.from_numpy(data).view(512, 512, c).permute(2, 0, 1)
        
 
        if c == 3:
            max_vals, _ = tensor.max(dim=0, keepdim=True)
            denom = torch.clamp(max_vals, min=1.0)
            tensor = tensor / denom
            
        return tensor.unsqueeze(0).to(device)


    base_path = os.path.join(train_folder, f"scene{scene_id}")
    
    print(f"Running inference on: {base_path}...")
    
    noisy = load_full_normalized(f"{base_path}_noisy.bin", 3)
    norm  = load_full_normalized(f"{base_path}_normal.bin", 3)
    depth = load_full_normalized(f"{base_path}_depth.bin", 1)
    

    x = torch.cat([noisy, norm, depth], dim=1)
    

    with torch.no_grad():
        output = model(x)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    

    save_name = f"scene{scene_id}_learned.png"
    plt.imsave(save_name, np.clip(output, 0, 1))
    print(f"Finished! Result saved as {save_name}")

if __name__ == "__main__":
    # train()

    test_one_scene("02", train_folder="train_data")