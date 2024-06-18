import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
from left_right_model import HemisphereStream, DualStream
import cv2, random
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms.functional import resize
import av
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


def create_model():
    model = VICReg()
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)  # Wrap the model for data parallelism
    return model

def setup_transforms():
    return transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(25, 25), sigma=(30, 30)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(contrast=0.8),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class LARS(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=1e-6, eta=0.001):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)
                
                # Compute the local learning rate for this layer
                local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm + 1e-8)
                
                # Update the momentum buffer
                if 'momentum_buffer' not in p.state:
                    buf = p.state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = p.state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=local_lr)
                
                # Update parameters
                p.data.add_(-lr, buf)

        return loss

# Ensure that all tensors and models are moved to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for computation.")

# Move models to the appropriate device
# Assuming models are defined in this file or imported, they should be moved to the device.
# Example: model.to(device)

# Ensure all tensors created in this file are automatically on the right device
# Example of creating a tensor on the device: tensor = torch.tensor(data, device=device)

config = {
    "data_dir": 'Charades_v1_480',
    "batch_size": 190,
    "learning_rate": 0.02,
    "epochs": 100,
    "num_workers": 8,
    "pin_memory": True,
    "drop_last": True,
    "world_size": torch.cuda.device_count(),  # Number of GPUs available
}


def save_checkpoint(model, path):
    """
    Save the model checkpoint.

    Args:
    - model: The model to save.
    - path: Path where the checkpoint will be saved.
    """
    torch.save(model.state_dict(), path)

def log_sampled_frames(frames, num_seq=1, seq_len=30, resize_shape=(68, 120)):
    """
    Log a grid of sampled frames from a video sequence to Weights & Biases (wandb).

    Args:
    - frames (torch.Tensor): A tensor of video frames of shape (num_seq, seq_len, C, H, W).
    - num_seq (int): Number of sequences sampled from the video.
    - seq_len (int): Number of frames in each sequence.
    - resize_shape (tuple): Resize shape for each frame, for consistent grid display.

    Raises:
    - ValueError: If the input tensor does not match the expected shape.
    """

    # Validate input tensor shape
    if not isinstance(frames, torch.Tensor) or len(frames.shape) != 5:
        raise ValueError("Frames must be a 5D tensor of shape (num_seq, seq_len, C, H, W).")
    if frames.shape[0] < num_seq or frames.shape[1] < seq_len:
        raise ValueError("Frames tensor does not have enough sequences or frames per sequence.")

    # Select the first frame from each sequence for simplicity
    selected_frames = frames[:, 0]  # This selects the first frame in each sequence


    # Resize frames for consistent display
    selected_frames_resized = torch.stack([resize(frame, resize_shape) for frame in selected_frames])

    # Create a grid of images
    frame_grid = make_grid(selected_frames_resized, nrow=num_seq, normalize=True)

    # Convert the tensor grid to a PIL image
    grid_image = to_pil_image(frame_grid)

    # Log the grid image to wandb
    wandb.log({"sampled_frames": [wandb.Image(grid_image, caption="Sampled Frames")]})

class TheDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_percentage=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mp4')]
        random.shuffle(self.video_files)
        num_files_to_use = int(len(self.video_files) * use_percentage)
        self.video_files = self.video_files[:num_files_to_use]

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_frames = read_video_frames(video_path, self.transform, seq_len=30, num_seq=1)
        return {'video': video_frames}

def read_video_frames(video_path, transform=None, num_seq=1, seq_len=30):
    container = av.open(video_path)
    stream = container.streams.video[0]

    frames = []
    #max_start_index = max(0, stream.frames - (num_seq * seq_len))
    max_start_index = 0
    start_index = random.randint(0, min(max_start_index, 210))
    end_index = start_index + num_seq * seq_len

    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count >= start_index and frame_count < end_index:
            img = frame.to_image()  # Convert to PIL Image
            if transform:
                img = transform(img)
            frames.append(img)
        frame_count += 1
        if frame_count >= end_index:
            break

    container.close()

    # Check if frames are already tensors, if not convert them
    if not isinstance(frames[0], torch.Tensor):
        frames = [to_tensor(frame) for frame in frames]

    # Ensure we have the correct number of frames
    if len(frames) != num_seq * seq_len:
        raise ValueError(f"Expected {num_seq * seq_len} frames, but got {len(frames)}")

    frames_tensor = torch.stack(frames, dim=0).view(num_seq, seq_len, 3, *frames[0].shape[1:])

    return frames_tensor

class VICReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 8000
        self.backbone = DualStream()

    def forward(self, x, mode):
        left_slow, left_fast, right_slow, right_fast = self.backbone(x)

        slow_repr_loss = F.mse_loss(left_slow, right_slow)
        fast_repr_loss = F.mse_loss(right_slow, right_fast)

        left_slow = left_slow - left_slow.mean(dim=0)
        right_slow = right_slow - right_slow.mean(dim=0)

        left_fast = left_fast - left_fast.mean(dim=0)
        right_fast = right_fast - right_fast.mean(dim=0)

        std_left_slow = torch.sqrt(left_slow.var(dim=0) + 0.0001)
        std_right_slow = torch.sqrt(right_slow.var(dim=0) + 0.0001)

        std_left_fast = torch.sqrt(left_fast.var(dim=0) + 0.0001)
        std_right_fast = torch.sqrt(right_fast.var(dim=0) + 0.0001)

        slow_std_loss = torch.mean(F.relu(1 - std_left_slow)) / 2 + torch.mean(F.relu(1 - std_right_slow)) / 2
        fast_std_loss = torch.mean(F.relu(1 - std_left_fast)) / 2 + torch.mean(F.relu(1 - std_right_fast)) / 2 

        cov_left_slow = (left_slow.T @ left_slow) / (config['batch_size'] - 1)
        cov_right_slow = (right_slow.T @ right_slow) / (config['batch_size'] - 1)
        slow_cov_loss = off_diagonal(cov_left_slow).pow_(2).sum().div(self.num_features) + off_diagonal(cov_right_slow).pow_(2).sum().div(self.num_features)

        cov_left_fast = (left_fast.T @ left_fast) / (config['batch_size'] - 1)
        cov_right_fast = (right_fast.T @ right_fast) / (config['batch_size'] - 1)
        fast_cov_loss = off_diagonal(cov_left_fast).pow_(2).sum().div(self.num_features) + off_diagonal(cov_right_fast).pow_(2).sum().div(self.num_features)
 
        slow_loss = 1 * slow_repr_loss + 4 * slow_std_loss.mean() + 0.04 * slow_cov_loss.mean()
        fast_loss = 1 * fast_repr_loss + 4 * fast_std_loss.mean() + 0.04 * fast_cov_loss.mean() 

        # Logging to wandb
        if mode == "train":
                    wandb.log({
            	        "slow_repr_loss": slow_repr_loss.item(),
                        "slow_std_loss": slow_std_loss.mean().item(),
                        "slow_cov_loss": slow_cov_loss.mean().item(),
                        "slow_total_loss": slow_loss.item()})
                    wandb.log({
            	        "fast_repr_loss": fast_repr_loss.item(),
                        "fast_std_loss": fast_std_loss.mean().item(),
                        "fast_cov_loss": fast_cov_loss.mean().item(),
                        "fast_total_loss": fast_loss.item()})
        if mode == "val":
                    wandb.log({
                        "repr_val": slow_repr_loss.item(),
                        "std_val": slow_std_loss.mean().item(),
                        "cov_val": slow_cov_loss.mean().item(),
                        "total_val": slow_loss.item()})
                    
                    wandb.log({
                        "repr_val": fast_repr_loss.item(),
                        "std_val": fast_std_loss.mean().item(),
                        "cov_val": fast_cov_loss.mean().item(),
                        "total_val": fast_loss.item()})

        return slow_loss + fast_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epoch, target_lr):
        self.optimizer = optimizer
        self.warmup_epoch = warmup_epoch
        self.target_lr = target_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epoch:
            lr = (self.target_lr / self.warmup_epoch) * self.current_epoch
        else:
            lr = self.target_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def train():
    model = create_model()
    wandb.init(project="left-right", config=config)

    # Setup transformations and data loaders
    transform = transforms.Compose([
        transforms.Resize((68, 120)),
        transforms.ToTensor(),
        #transforms.GaussianBlur(kernel_size=(9, 9), sigma=(3, 3)),
        #transforms.RandomApply([
            #transforms.Grayscale(num_output_channels=3)
        #], p=0.7),
        #transforms.ColorJitter(contrast=0.3, hue=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = TheDataset(root_dir=config['data_dir'], transform=transform)
    train_size = int(0.80 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_filenames = [full_dataset.video_files[idx] for idx in train_dataset.indices]
    val_filenames = [full_dataset.video_files[idx] for idx in val_dataset.indices]

    with open('train_files.txt', 'w') as f:
        for filename in train_filenames:
            f.write(f"{filename}\n")

    with open('val_files.txt', 'w') as f:
        for filename in val_filenames:
            f.write(f"{filename}\n")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=config['drop_last'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=config['drop_last'])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    #optimizer = LARS(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-6, eta=0.0002)

    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - 6, eta_min=2e-6)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epoch=6, target_lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            inputs = batch['video']
            optimizer.zero_grad()
            loss = model(inputs, "train")
            loss = loss.mean()
            print(loss)
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1.0)
            if epoch < 6:
                warmup_scheduler.step()
                optimizer.step()
                #scheduler.step()
            else:
                scheduler.step()
                optimizer.step()
            #optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()

            print(f"Train Epoch: {epoch} [{i * len(inputs)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            wandb.log({"batch_loss": loss.item()})
            log_sampled_frames(inputs[0])

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch} Average Loss: {average_loss}')
        wandb.log({"epoch_average_loss": average_loss, "epoch": epoch})

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs = batch['video']
                loss = model(inputs, "val").mean()
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)

        print(f'Validation Loss: {average_val_loss}')
        wandb.log({"validation_loss": average_val_loss, "epoch": epoch})

        if (epoch + 1) % 1 == 0:
            checkpoint_path = f'model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    cleanup()

if __name__ == '__main__':
    train()
