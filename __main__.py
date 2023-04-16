import openai
import open_clip
import torch
import os
from PIL import Image
import cv2
import json
from Captions import WatchVideo
from json_processing import process_video, write_output_to_file, read_captions_from_file
from VID2AUD import vid2aud
from openai_chat import generate_openai_chat_response
from audio_environmental_sounds import lemme_see
from audio_transcript import transcribe_audio
from audio_song import shazam
import time
import asyncio


VIDEO = "/Users/jimvincentwagner/tests/Y2Mate.is - talk&text  win vs ginebra gmae 4.#video#shortsfeed#amazing#satisfying#basketball--PtIs03wULc-720p-1657302317151.mp4"
SCAN_FPS = 50
CLASSES_N = 3



async def main():
    start_time = time.time()

    # Define the video path for downstream processing
    video_path = VIDEO

    # Process the video witch x fps and generate captions and save them to a JSON file
    Captions = WatchVideo(video_path, SCAN_FPS)

    # Load the captions from the JSON file
    txt = read_captions_from_file("output.json")

    # Combine the captions into a single string
    prompt_lines = "\n".join(txt)

    # MoviePy for audio extraction
    audio = vid2aud(video_path)

    # Get the audio environmental transcription via zac script
    audio_info = lemme_see(lemme_see, top_n= CLASSES_N)  

    # Transcribe Text to Speech
    transcribed_audio = transcribe_audio(audio)

    # Shazam the audio
    shazam_audio = await shazam(audio)
    '''if shazam_audio == "No song found":
        title = "No song found"
    else:
        title = shazam_audio['track']['title']'''
    print(shazam_audio)

    # Check the audio transcription
    print("Subtitles:" + transcribed_audio)

    # Generate the response from OpenAI Chat Completion
    response = generate_openai_chat_response(prompt_lines, str(audio_info), str(transcribed_audio))

    # Print the response
    print(response)

    os.remove("output.json")
    os.remove("output.mp3")

    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))

if __name__ == "__main__":
    asyncio.run(main())