import openai

def generate_openai_chat_response(prompt_lines, aud_info):
    # Load the OpenAI API key
    openai.api_key = "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"

    # Set up OpenAI Chat Completion API parameters
    MODEL = "gpt-3.5-turbo-0301"

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are aiding in categorizing Videos."},
            {"role": "user", "content": "The following Information are I. Lines are the captions that Computer Vision Information Retrieval model outputs and II. The output of Audio Classification System thack recognizes Environmental sounds. The Intention is to categorize and label the video. Provide 5 hashtags for it. Just the regular ones of a social media app. Also Place it into one of the categories and give it the Label: 1. Sports , 2. User-generated content, Private Event, 5. Outside with people, 6. inside of the appartment . Explain."},
            {"role": "assistant", "content": "Okay, so what are the captions?"},
            {"role": "user", "content": "Captions:\n\n" + prompt_lines},
            {"role": "system", "content": "And what does the Audio Classification System recognize?"},
            {"role": "user", "content": "Audio Classification System:" + aud_info},
        ],
        temperature=0,
    )

    # Return the response from OpenAI Chat Completion
    return response['choices'][0]['message']['content']
