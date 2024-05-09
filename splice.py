import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

def get_video_dimensions(video_path):
    """Retrieve the dimensions of the video."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dimensions = result.stdout.strip()
    return dimensions

def splice_video(input_path, output_dir, segment_length, first_video_dimensions):
    """Function to splice a single video if dimensions match."""
    # Check if the video dimensions match the first video
    if get_video_dimensions(input_path) != first_video_dimensions:
        print(f"Skipping {os.path.basename(input_path)} due to dimension mismatch.")
        return

    base_output = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_part_')
    cmd_duration = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())
    full_segments = int(duration // segment_length)

    command = [
        'ffmpeg',
        '-i', input_path,
        '-c', 'copy',
        '-map', '0',
        '-segment_time', str(segment_length),
        '-t', str(full_segments * segment_length),
        '-f', 'segment',
        '-reset_timestamps', '1',
        base_output + '%03d.mp4'
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Processed {os.path.basename(input_path)}")

def chop_videos(input_dir, output_dir, segment_length=8):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mp4")]
    if not video_files:
        print("No MP4 files found in the directory.")
        return

    first_video_path = video_files[0]
    first_video_dimensions = get_video_dimensions(first_video_path)

    # Use ProcessPoolExecutor to parallelize video processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(splice_video, video_path, output_dir, segment_length, first_video_dimensions)
                   for video_path in video_files]
        for future in futures:
            future.result()  # Wait for all futures to complete

input_directory = '/home/libiadm/export/HDD2/datasets/EGO4D/v2/clips'
output_directory = 'splice'
chop_videos(input_directory, output_directory)

