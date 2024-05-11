import torch
from model import DualStream
import numpy as np

def generate_dummy_video(batch_size, num_sequences, sequence_length, channels, height, width):
    """
    Generate a dummy video tensor with random data.
    
    Args:
    - batch_size (int): Number of videos in the batch.
    - num_sequences (int): Number of sequences per video.
    - sequence_length (int): Number of frames per sequence.
    - channels (int): Number of channels per frame (e.g., 3 for RGB).
    - height (int): Frame height.
    - width (int): Frame width.
    
    Returns:
    - torch.Tensor: A tensor representing the batch of videos.
    """
    # Create a random tensor to simulate video data
    return torch.rand(batch_size, num_sequences, sequence_length, channels, height, width)

def test_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = DualStream()
    model.to(device)
    
    # Generate dummy video data
    # Assuming the input dimensions expected by the model are:
    # Batch size: 1, Num sequences: 1, Sequence length: 3, Channels: 3, Height: 180, Width: 240
    dummy_video = generate_dummy_video(batch_size=1, num_sequences=8, sequence_length=30, channels=3, height=240, width=180)
    dummy_video = dummy_video.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        prediction, target, concat_layer, future_context = model(dummy_video)
    
    # Print outputs
    print("Prediction:", prediction.shape)
    print("Target:", target.shape)
    print("Concatenated Layer Output:", concat_layer.shape)
    print("Future Context:", future_context.shape)

if __name__ == '__main__':
    test_model()
