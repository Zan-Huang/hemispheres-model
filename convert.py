import os
import av
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def process_video(video_path):
    """
    Process a single video file to extract frames and return them as a list of numpy arrays.
    
    Args:
    video_path (str): Path to the video file.
    
    Returns:
    tuple: Filename and list of numpy arrays representing the frames.
    """
    filename = os.path.basename(video_path)
    print(f"Processing {filename}...")
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []

    for frame in container.decode(stream):
        img = frame.to_image()
        img_array = np.array(img)
        frames.append(img_array)

    container.close()
    print(f"Finished processing {filename}")
    return (filename, frames)

def video_to_h5(input_dir, output_dir, output_file):
    """
    Convert all videos in the input directory to an HDF5 file and store it in the output directory using parallel processing.
    
    Args:
    input_dir (str): Directory containing video files.
    output_dir (str): Directory to store the output HDF5 file.
    output_file (str): Name of the output HDF5 file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the output HDF5 file
    output_path = os.path.join(output_dir, output_file)
    
    # List all video files
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5f:
        with ProcessPoolExecutor() as executor:
            # Process each video file in parallel
            results = executor.map(process_video, video_files)
            
            for filename, frames in results:
                # Create a group for each video file
                grp = h5f.create_group(filename)
                
                # Store each frame as a separate dataset within the group
                for i, frame_data in enumerate(frames):
                    grp.create_dataset(f'frame_{i}', data=frame_data, compression="gzip")

if __name__ == "__main__":
    input_directory = 'splice'  # Directory containing videos
    output_directory = 'splice_convert'  # Directory to store the converted HDF5 file
    output_h5_file = 'video_frames.h5'  # Output HDF5 file name
    video_to_h5(input_directory, output_directory, output_h5_file)
    print("All videos have been processed and saved to HDF5 in the 'splice_convert' directory.")
    print("All videos have been processed and saved to HDF5 in the 'splice_convert' directory.")
