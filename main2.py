import os
import argparse
import json
import multiprocessing
from pathlib import Path

import open_clip
import torch
from PIL import Image
import cv2

from Captions import WatchVideo


def main():
    parser = argparse.ArgumentParser(description='Extract captions from videos')
    parser.add_argument('input', type=str, help='Path to input directory containing videos')
    parser.add_argument('output', type=str, help='Path to output directory for storing JSON files')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to skip between each processed frame')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of videos to process in parallel')
    args = parser.parse_args()

    # Load the model
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

    # Create the output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of video paths from the input directory
    video_paths = list(Path(args.input).glob('*.mp4'))

    # Define the function to process a single video
    def process_video(video_path):
        captions, fps, width, height, fourcc = WatchVideo(video_path, args.frames, model, transform)
        output_dict = {"captions": captions, "fps": fps, "width": width, "height": height, "fourcc": fourcc}
        output_path = output_dir / (video_path.stem + '.json')
        with open(output_path, 'w') as f:
            json.dump(output_dict, f)

    # Create a multiprocessing pool to process videos in parallel
    with multiprocessing.Pool(processes=args.batch_size) as pool:
        pool.map(process_video, video_paths)




if __name__ == '__main__':
    main()
