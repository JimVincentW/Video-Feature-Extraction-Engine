import whisper

def transcribe_audio(audio):
    model = whisper.load_model("large")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("output.mp3")
    result = model.transcribe(audio, fp16=False)
    subtitles = result["text"]
    
    # if item is NoneType
    if subtitles is None:
        return "No subtitles found"
    else:
        return subtitles