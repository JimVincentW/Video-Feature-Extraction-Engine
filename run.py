import argparse
import openai
import open_clip
import torch
from PIL import Image
import cv2
import json
from Captions import WatchVideo
from json_processing import process_video, write_output_to_file, read_captions_from_file
from VID2AUD import vid2aud
from openai_chat import generate_openai_chat_response
from audio_environmental_sounds import lemme_see
from audio_transcript import transcribe_audio
import os


def main(video_path, scan_fps, classes_n, save_results):
    # Process the video with x fps and generate captions and save them to a JSON file
    Captions = WatchVideo(video_path, scan_fps)

    # Load the captions from the JSON file
    txt = read_captions_from_file("output.json")

    # Combine the captions into a single string
    prompt_lines = "\n".join(txt)

    # MoviePy for audio extraction
    audio = vid2aud(video_path)

    # Get the audio environmental transcription via zac script
    audio_info = lemme_see(lemme_see, top_n=classes_n)

    # Transcribe Text to Speech
    transcribed_audio = transcribe_audio(audio)

    # Check the audio transcription
    print(audio_info)

    # Generate the response from OpenAI Chat Completion
    response = generate_openai_chat_response(prompt_lines, str(audio_info), str(transcribed_audio))

    # Print the response
    print(response)

    # Save the results to a file if the --save-results argument is provided
    if save_results:
        with open("results.txt", "w") as f:
            f.write(response)

    # Delete the output JSON and MP3 files
    os.remove("output.json")
    os.remove("output.mp3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video and generate response from OpenAI Chat')
    parser.add_argument('--video', type=str, help='Path to input video file', required=True)
    parser.add_argument('--scan-fps', type=int, help='FPS for video processing', default=50)
    parser.add_argument('--classes-n', type=int, help='Number of environmental sounds to detect', default=3)
    parser.add_argument('--save-results', dest='save_results', action='store_true', help='Save the results to a file')

    args = parser.parse_args()

    main(args.video, args.scan_fps, args.classes_n, args.save_results)
