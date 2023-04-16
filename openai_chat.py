import openai

def generate_openai_chat_response(prompt_lines, audio_info, transcribed_audio):
    # Load the OpenAI API key
    openai.api_key = "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"

    # Set up OpenAI Chat Completion API parameters
    MODEL = "gpt-3.5-turbo-0301"

    labels =  "1. Professional Narated Content , 2. User-remixed content, Private Event, 5. Outside video with normal people, 6. inside of the appartment"


    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are aiding in categorizing Videos."},
            {"role": "user", "content": "The following Information are I. Lines are the captions that Computer Vision Information Retrieval model outputs, II. The output of Audio Classification System thack recognizes Environmental sounds and III. a transcription attempt if there is normal voice. The Intention is to categorize and label the video. If one information is missing or too random, select the next best option. Provide 5 hashtags for it. Just the regular ones of a social media app. Also Place it into one of the categories and give it the Label:" + labels + " . Explain."},
            {"role": "assistant", "content": "Okay, so what are the captions?"},
            {"role": "user", "content": "Captions:\n\n" + prompt_lines},
            {"role": "assistant", "content": "And what does the Audio Classification System recognize?"},
            {"role": "user", "content": "Audio Classification System:" + audio_info},
            {"role": "assistant", "content": "And what does the Transcription Attempt recognize?"},
            {"role": "user", "content": "Transcription Attempt:" + transcribed_audio + ". If there is not Subtitles. Categorize it without this information."},
        ],
        temperature=0,
    )

    # Return the response from OpenAI Chat Completion
    return response['choices'][0]['message']['content']
