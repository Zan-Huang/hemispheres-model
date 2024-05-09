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


def save_checkpoint(model, path):
    """
    Save the model checkpoint.

    Args:
    - model: The model to save.
    - path: Path where the checkpoint will be saved.
    """
    torch.save(model.state_dict(), path)

def log_sampled_frames(frames, num_seq=8, seq_len=5, downsample=3, resize_shape=(128, 128)):
    """
    Log a grid of sampled frames from a video sequence.

    Args:
    - frames: A tensor of video frames of shape (num_seq, seq_len, C, H, W).
    - num_seq: Number of sequences sampled from the video.
    - seq_len: Number of frames in each sequence.
    - downsample: Factor by which frames were downsampled.
    - resize_shape: Resize shape for each frame, for consistent grid display.
    """
    selected_frames = frames[:, 0]  # Select the first frame from each sequence for simplicity
    selected_frames_resized = torch.stack([TF.resize(frame, resize_shape) for frame in selected_frames])
    frame_grid = make_grid(selected_frames_resized, nrow=num_seq, normalize=True)
    grid_image = to_pil_image(frame_grid)
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

def read_video_frames(video_path, transform, seq_len=30, num_seq=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_indices = []

    # Calculate the spacing between the starts of each sequence
    spacing = total_frames // num_seq

    for seq_index in range(num_seq):
        start_frame = seq_index * spacing
        for frame_index in range(seq_len):
            frame_indices.append(start_frame + frame_index)

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames, dim=0).view(num_seq, seq_len, *frames[0].size())

# Configuration
config = {
    "data_dir": 'splice',
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 2,
    "learning_rate": 0.001,
    "epochs": 100,
    "num_workers": 16,
    "pin_memory": True,
    "drop_last": True
}

def setup_transforms():
    return transforms.Compose([
        transforms.Resize((240, 180)),
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
    unique_step_identifier = 0
    for i, batch in enumerate(train_loader):
        inputs = batch['video'].to(device)
        score, mask, embedding, future_context = model(inputs)
        target, (_, B2, NS, NP, SQ) = process_output(mask)
        score_flattened = score.reshape(-1, B2*NS*SQ)
        target_flattened = target.reshape(-1, B2*NS*SQ).int().argmax(dim=1)
        loss = criterion(score_flattened, target_flattened)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']}, step=unique_step_identifier)
        unique_step_identifier += 1

        if i % 100 == 0:
            log_sampled_frames(inputs[0], num_seq=8, seq_len=5, downsample=3)
            wandb.log({"top15_accuracy": calc_topk_accuracy(score_flattened, target_flattened, topk=(1,5))[0]}, step=unique_step_identifier)

        if i % 500 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=unique_step_identifier)

    return total_loss / len(train_loader)

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
            total_top_k_accuracy += calc_topk_accuracy(score_flattened, target_flattened, topk=(1,3))[0]
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

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], epoch)
        val_loss, val_top_k_accuracy = validate(model, val_loader, criterion, config['device'])
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top_k_accuracy": val_top_k_accuracy})

        # Save model checkpoint
        if (epoch + 1) % 1 == 0:  # Adjust as per your checkpoint saving frequency
            checkpoint_path = os.path.join("models", f'model_epoch_{epoch+1}.pth')
            save_checkpoint(model.module if isinstance(model, nn.DataParallel) else model, checkpoint_path)
            print(f"Saved model checkpoint at {checkpoint_path}")

    wandb.finish()

if __name__ == '__main__':
    main()
