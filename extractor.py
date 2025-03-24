import cv2
import os
import numpy as np
import pandas as pd
import subprocess
import csv


def main():
    # clear out output stuff
    dir_name = 'output_frames'
    if os.path.exists(dir_name):
        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name,filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}.")


    frames_with_info = extract()
    features = []
    for frame, timestamp, frame_filename in frames_with_info:
        hist = extract_colour_histogram(frame=frame)
        edges = extract_edge_features(frame=frame)
        combined = [timestamp, frame_filename] + hist.tolist() + edges.tolist()
        features.append(combined)
    print("Feature data has been collected!")
    hist_size = 8*8*8 # 8 bins per channel
    edge_size = 1000 # data is truncated to 1000 values
  
    print("Writing data to csv")
    save_features_streaming(frames_with_info=frames_with_info, hist_size=hist_size, edge_size=edge_size, output_file='output_data/output_features.csv')
    print("Extraction completed!")
    
    
def save_features_streaming(frames_with_info, hist_size, edge_size, output_file):
    hist_columns = [f'hist_bin_{i}' for i in range(hist_size)]
    edge_columns = [f'edge_{i}' for i in range(edge_size)]
    column_names = ['timestamp', 'frame_filename'] + hist_columns + edge_columns

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)  # Write header

        for frame, timestamp, filename in frames_with_info:
            hist = extract_colour_histogram(frame)
            edges = extract_edge_features(frame)
            combined = [timestamp, filename] + hist.tolist() + edges.tolist()
            writer.writerow(combined)

    print(f"Features saved to {output_file}")

def extract():
    video_path = 'input_videos/input.mp4'
    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    extract_audio(video_path, 'output_data/audio.mp3')
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) # timestamp in milliseconds
        frame_filename = f"{output_dir}/frame_{frame_count}_time_{int(timestamp)}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_data = (frame, int(timestamp), frame_filename)
        frames.append(frame_data)
        frame_count = len(frames)

    cap.release()
    print(f"Extracted {frame_count} frames.")
    return frames

def extract_colour_histogram(frame):
    hist = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]) # 8 bins per channel
    hist = cv2.normalize(hist,hist).flatten()
    return hist

def extract_edge_features(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # greyscale it
    edges = cv2.Canny(grey,100,200)
    return edges.flatten()[:1000] # truncate to keep it simple

def extract_audio(input_video, output_audio):
    command = ['ffmpeg', '-i', input_video, '-q:a', '0', '-map', 'a', output_audio, '-y']
    subprocess.run(command, check=True)
    print(f"Audio extracted to {output_audio}")

def save_features_to_csv(features, hist_size, edge_size, output_file):
    #column names
    hist_columns = [f'hist_bin_{i}' for i in range(hist_size)]
    edge_columns = [f'edge+{i}' for i in range(edge_size)]
    column_names = ['timestamp','frame_filename'] + hist_columns + edge_columns

    df = pd.DataFrame(features, columns=column_names)
    df.to_csv(output_file, index=False)
    print(f"Saved combined features to {output_file}")

if __name__ == '__main__':
    main()