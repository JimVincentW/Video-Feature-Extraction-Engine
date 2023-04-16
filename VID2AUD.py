from moviepy.editor import *

def vid2aud(video_path):
    output_audio_path = "output.mp3"

    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, verbose=False, logger=None)

    return output_audio_path
