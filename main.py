import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
from model import DualStream
import cv2, random
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms.functional import resize
import av
from torch.optim.lr_scheduler import ReduceLROnPlateau


def save_checkpoint(model, path):
    """
    Save the model checkpoint.

    Args:
    - model: The model to save.
    - path: Path where the checkpoint will be saved.
    """
    torch.save(model.state_dict(), path)

def log_sampled_frames(frames, num_seq=8, seq_len=30, resize_shape=(90, 120)):
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

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res

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
        video_frames = read_video_frames(video_path, self.transform, seq_len=30, num_seq=8)
        return {'video': video_frames}

def read_video_frames(video_path, transform=None, num_seq=8, seq_len=30):
    container = av.open(video_path)
    stream = container.streams.video[0]

    frames = []
    frame_indices = list(range(0, num_seq * seq_len))

    for frame in container.decode(video=0):
        img = frame.to_image()  # Convert to PIL Image
        if transform:
            img = transform(img)
        frames.append(img)

    container.close()

    # Check if frames are already tensors, if not convert them
    if not isinstance(frames[0], torch.Tensor):
        frames = [to_tensor(frame) for frame in frames]

    # Ensure we have the correct number of frames
    if len(frames) != num_seq * seq_len:
        raise ValueError(f"Expected {num_seq * seq_len} frames, but got {len(frames)}")

    frames_tensor = torch.stack(frames, dim=0).view(num_seq, seq_len, 3, *frames[0].shape[1:])

    return frames_tensor
# Configuration
config = {
    "data_dir": 'splice',
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 12,
    "learning_rate": 0.001,
    "epochs": 100,
    "num_workers": 5,
    "pin_memory": True,
    "drop_last": True
}

def setup_transforms():
    return transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.ToTensor(),
    ])

def setup_data_loaders(data_dir, transform):
    full_dataset = TheDataset(root_dir=data_dir, transform=transform)

    train_size = int(0.85 * len(full_dataset))
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
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_hemisphere_loss = 0
    unique_step_identifier = 0
    for i, batch in enumerate(train_loader):
        inputs = batch['video'].to(device)
        score, mask, embedding, future_context, cosine_score = model(inputs)
        target, (_, B2, NS, NP, SQ) = process_output(mask)
        score_flattened = score.reshape(-1, B2*NS*SQ)
        target_flattened = target.reshape(-1, B2*NS*SQ).int().argmax(dim=1)
        loss = criterion(score_flattened, target_flattened) + cosine_score.mean()
        
        print(f"Step {unique_step_identifier}: total loss: {loss.item()}")
        print(f"Step {unique_step_identifier}: hemisphere loss: {cosine_score.mean().item()}")
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_hemisphere_loss += cosine_score.mean().item()
        wandb.log({"train_loss": loss.item(), "hemisphere_loss": cosine_score.mean().item(), "learning_rate": optimizer.param_groups[0]['lr']}, step=unique_step_identifier)
        unique_step_identifier += 1

        if i % 25 == 0:
            log_sampled_frames(inputs[0], num_seq=8, seq_len=30)
            wandb.log({"top15_accuracy": calc_topk_accuracy(score_flattened, target_flattened, topk=(1,5))[0]}, step=unique_step_identifier)

        if i % 250 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=unique_step_identifier)

    return total_loss / len(train_loader), total_hemisphere_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_top_k_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['video'].to(device)
            score, mask, embedding, future_context = model(inputs)
            target, (_, B2, NS, NP, SQ) = process_output(mask)
            score_flattened = score.reshape(-1, B2*NS*SQ)
            target_flattened = target.reshape(-1, B2*NS*SQ).int().argmax(dim=1)
            loss = criterion(score_flattened, target_flattened)
            total_loss += loss.item()
            total_top_k_accuracy += calc_topk_accuracy(score_flattened, target_flattened, topk=(1,5))[0]
    average_val_loss = total_loss / len(val_loader)
    average_val_top_k_accuracy = total_top_k_accuracy / len(val_loader)
    wandb.log({"val_loss": average_val_loss, "val_top_k_accuracy": average_val_top_k_accuracy})

def main():
    wandb.init(project="left-right", config=config)
    transform = setup_transforms()
    train_loader, val_loader = setup_data_loaders(config['data_dir'], transform)
    
    # Initialize the model and move it to GPU
    model = DualStream()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model for multi-GPU training
        model = nn.DataParallel(model)
    model.to(config['device'])
    
    criterion = nn.CrossEntropyLoss().to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(config['epochs']):
        train_loss, train_hemisphere_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], epoch)
        val_loss, val_top_k_accuracy = validate(model, val_loader, criterion, config['device'])
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_hemisphere_loss": train_hemisphere_loss,
            "val_loss": val_loss,
            "val_top_k_accuracy": val_top_k_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join("models", f'model_epoch_{epoch+1}.pth')
            save_checkpoint(model.module if isinstance(model, nn.DataParallel) else model, checkpoint_path)
            print(f"Saved model checkpoint at {checkpoint_path}")

    wandb.finish()

if __name__ == '__main__':
    main()
