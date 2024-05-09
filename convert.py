import os
import av
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_video(video_path):
    try:
        logging.info(f"Starting processing of {video_path}")
        filename = os.path.basename(video_path)
        container = av.open(video_path)
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            img = frame.to_image()
            img = img.resize((240, 180))
            img_array = np.array(img)
            frames.append(img_array)
        container.close()
        logging.info(f"Finished processing {filename}")
        return filename, frames
    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return None

def video_to_h5(input_dir, output_dir, output_file, batch_size=4):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    with h5py.File(output_path, 'w') as h5f:
        for i in range(0, len(video_files), batch_size):
            with ProcessPoolExecutor(max_workers=4) as executor:
                batch_files = video_files[i:i+batch_size]
                futures = {executor.submit(process_video, video): video for video in batch_files}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        filename, frames = result
                        grp = h5f.create_group(filename)
                        for j, frame_data in enumerate(frames):
                            grp.create_dataset(f'frame_{j}', data=frame_data, compression="gzip")

if __name__ == "__main__":
    video_to_h5('splice', 'splice_convert', 'video_frames.h5')
    logging.info("All videos have been processed and saved to HDF5.")
