import os
import torch
from torchvision.datasets.utils import download_url
from torchvision.datasets import UCF101
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from model import DualStream
import ssl
from PIL import Image
import cv2
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import random
import wandb
import torch.nn as nn
from torchvision.transforms import Normalize, RandomGrayscale, RandomHorizontalFlip, ColorJitter
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import v2

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms

data_dir = '/home/libiadm/export/HDD2/datasets/moments_in_time/Moments_in_Time_Raw'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ssl._create_default_https_context = ssl._create_unverified_context


criterion = nn.CrossEntropyLoss().to(device)

"""transform = Compose([
    Resize((128, 128)),
    #Grayscale(),
    ToTensor(),
    #RandomHorizontalFlip(),
    #ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    #Normalize(mean=[0.5], std=[0.5]),
])"""

transform = v2.Compose([
    #v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.Resize((256, 340)),
    v2.ToTensor(),
])

class MomentsInTimeDataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None, use_percentage=1.0, seq_len=8, num_seq=8, downsample=3):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.video_files = []
        
        # Adjust for Moments in Time directory structure
        for action_category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, action_category)
            if os.path.isdir(category_path):
                for video_file in os.listdir(category_path):
                    video_path = os.path.join(category_path, video_file)
                    # Adjust the file extension as needed for your dataset
                    if video_file.endswith('.mp4'):
                        self.video_files.append(video_path)
        
        random.shuffle(self.video_files)
        num_files_to_use = int(len(self.video_files) * use_percentage)
        self.video_files = self.video_files[:num_files_to_use]

    def __len__(self):
        print(len(self.video_files))
        return len(self.video_files)

    def __getitem__(self, idx):
        video_frames = None
        attempts = 0
        while video_frames is None:
            video_path = self.video_files[(idx + attempts) % len(self.video_files)]
            video_frames = read_video_frames(video_path, self.transform, self.seq_len, self.num_seq, self.downsample)
            attempts += 1
            if attempts >= len(self.video_files):
                raise RuntimeError("Failed to find a video with enough frames")
        return {'video': video_frames}

def read_video_frames(video_path, transform, seq_len=8, num_seq=8, downsample=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = []

    if total_frames > 0:
        spacing = max(1, (total_frames - downsample * (seq_len - 1)) // num_seq)

    for seq_index in range(num_seq):
        start_frame = seq_index * spacing
        for frame_index in range(seq_len):
            if start_frame + frame_index * downsample < total_frames:
                frame_indices.append(start_frame + frame_index * downsample)
            else:
                # Not enough frames
                cap.release()
                return None  # Indicate insufficient frames

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = Image.fromarray(frame)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames, dim=0).view(num_seq, seq_len, *frames[0].size())


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

BATCH_SIZE = 2
LR = 0.00015

def main():
    wandb.init(project="Dual-Stream-New", config = {"learning_rate": LR, "epochs": 100, "batch_size": BATCH_SIZE, "architecture": "Dual-Stream"})
    
    train_dataset = MomentsInTimeDataset(root_dir=data_dir, split='training', transform=transform)
    val_dataset = MomentsInTimeDataset(root_dir=data_dir, split='validation', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=True)
    
    model = DualStream()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    inputs = next(iter(train_loader))['video'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5, amsgrad=True, eps=1e-8)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.80)

    num_epochs = 100

    unique_step_identifier = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = batch['video'].to(device)
            #inputs = torch.zeros(inputs.shape).to(device)
            #inputs = single_example

            score, mask = model(inputs)

            B = inputs.size(0)

            target, (_, B2, NS, NP, SQ) = process_output(mask)

            score_flattened = score.reshape(B*NP*SQ, B2*NS*SQ)
            target_flattened = target.reshape(B*NP*SQ, B2*NS*SQ)

            target_flattened= target_flattened.int()
            target_flattened = target_flattened.argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            
            running_loss += loss.item()

            print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')
            unique_step_identifier += 1
            wandb.log({"train_loss": loss.item()}, step=unique_step_identifier)
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=unique_step_identifier)

            if i % 10 == 0:
                input_frame_to_log = inputs[0, :, 0, :, :].cpu()
                wandb.log({"example_input": [wandb.Image(input_frame_to_log, caption="Example Input Frame")]}, step=unique_step_identifier)

            wandb.log({"top15_accuracy": calc_topk_accuracy(score_flattened, target_flattened, topk=(1,5))[0]}, step=unique_step_identifier)

            optimizer.zero_grad()
            loss.backward()

            if(i % 500 == 0):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=unique_step_identifier)

            optimizer.step()

        #scheduler.step()
        model.eval()
        val_loss = 0.0
        val_top_k_accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['video'].to(device)

                score, mask = model(inputs)
                B = inputs.size(0)
                
                score_flattened = score.reshape(B*NP*SQ, B2*NS*SQ)

                target_flattened = target.reshape(B*NP*SQ, B2*NS*SQ)
                target_flattened_numerical = target_flattened.int()
                target_flattened = target_flattened_numerical.argmax(dim=1)


                loss = criterion(score_flattened, target_flattened)
                val_loss += loss.item()
                val_top_k_accuracy += calc_topk_accuracy(score_flattened, target_flattened, topk=(1,3))[0]

        average_val_loss = val_loss / len(val_loader)
        average_val_top_k_accuracy = val_top_k_accuracy / len(val_loader)
        wandb.log({"val_loss": average_val_loss})
        wandb.log({"validation_top15_accuracy": average_val_top_k_accuracy})

        checkpoint_path = 'model.pth'
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Saved model checkpoint at epoch {epoch + 1}")

    wandb.finish()

if __name__ == '__main__':
    main()
