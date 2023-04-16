import json
import torch
import torch.nn.functional as F
import whisper
from whisper.audio import N_FRAMES, N_MELS, log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, Tokenizer, get_tokenizer

def classify_audio(model, audio_path, class_names, tokenizer, top_n=1, internal_lm_average_logprobs=None, verbose=False):
    audio_features = calculate_audio_features(audio_path, model)
    average_logprobs = calculate_average_logprobs(model, audio_features, class_names, tokenizer)

    if internal_lm_average_logprobs is not None:
        average_logprobs -= internal_lm_average_logprobs

    sorted_indices = sorted(range(len(class_names)), key=lambda i: average_logprobs[i], reverse=True)

    if verbose:
        print("  Average log probabilities for each class:")
        for i in sorted_indices:
            print(f"    {class_names[i]}: {average_logprobs[i]:.3f}")

    # Return the top N classes
    top_classes = [(class_names[i], average_logprobs[i].item()) for i in sorted_indices[:top_n]]
    return top_classes

def calculate_audio_features(audio_path, model):
    mel = log_mel_spectrogram(audio_path)
    segment = pad_or_trim(mel, N_FRAMES).to(model.device)
    return model.embed_audio(segment.unsqueeze(0))

def calculate_average_logprobs(model, audio_features, class_names, tokenizer):
    initial_tokens = (
        torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(0).to(model.device)
    )
    eot_token = torch.tensor([tokenizer.eot]).unsqueeze(0).to(model.device)

    average_logprobs = torch.zeros(len(class_names))

    for i, class_name in enumerate(class_names):
        class_name_tokens = (
            torch.tensor(tokenizer.encode(" " + class_name)).unsqueeze(0).to(model.device)
        )
        input_tokens = torch.cat([initial_tokens, class_name_tokens, eot_token], dim=1)

        logits = model.logits(input_tokens, audio_features)
        logprobs = F.log_softmax(logits, dim=-1).squeeze(0)
        logprobs = logprobs[len(tokenizer.sot_sequence_including_notimestamps) - 1 : -1]
        logprobs = torch.gather(logprobs, dim=-1, index=class_name_tokens.view(-1, 1))
        average_logprob = logprobs.mean().item()
        average_logprobs[i] = average_logprob

    return average_logprobs


def lemme_see(audio_path, top_n=1):
    # Load the Whisper model and tokenizer
    model_name = "large"  # or other available models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device)
    tokenizer = get_tokenizer(multilingual=".en" not in model_name, language="en")

    # Set the audio file path and class names
    audio_path = "output.mp3"
    class_names = [
    "cat",
    "talking",
    "conversational",
    "cheering",
    "singing",
    "music",
    "crowd",
    "fireworks",
    "wind",
    "cow",
    "sea_waves",
    "thunderstorm",
    "keyboard_typing",
    "water_drops",
    "engine",
    "frog",
    "rain",
    "stadium",
    "cheering",
    "insects",
    "sheep",
    "sneezing",
    "door_wood_creaks",
    "chainsaw",
    "pig",
    "crackling_fire",
    "vacuum_cleaner",
    "laughing",
    "chirping_birds",
    "brushing_teeth",
    "car_horn",
    "can_opening",
    "breathing",
    "crow",
    "siren",
    "airplane",
    "crickets",
    "snoring",
    "train",
    "dog",
    "drinking_sipping",
    "hen",
    "clock_tick",
    "church_bells",
    "hand_saw",
    "clock_alarm",
    "crying_baby",
    "toilet_flush",
    "clapping",
    "helicopter",
    "pouring_water",
    "coughing",
    "glass_breaking",
    "mouse_click",
    "washing_machine",
    "door_wood_knock",
    "footsteps",
    "rooster",
]

    # Classify the audio
    result = classify_audio(model, audio_path, class_names, tokenizer, top_n=top_n, verbose=False)

    # Return the result
    return result