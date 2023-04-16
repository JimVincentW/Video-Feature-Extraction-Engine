from moviepy.editor import *
import os


def vid2aud(video_path):
    # Check if the input video file exists
    if not os.path.exists(video_path):
        print("Error: video file not found")
        return None

    output_audio_path = "output.mp3"

    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, verbose=False, logger=None)

        return output_audio_path
    except Exception as e:
        print(f"Error: {e}")
        return "No Audio"
