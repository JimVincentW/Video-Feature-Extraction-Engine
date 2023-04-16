import cv2
from PIL import Image
import torch
import open_clip
import json

def process_video(video_path):
    # Load the CLIP model
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an empty list to store the output captions
    captions = []

    # Loop through every 5th frame of the video
    for i in range(0, total_frames, 5):
        # Process the frame and generate caption
        caption = process_frame(cap, i, model, transform, total_frames)
        captions.append(caption)

    # Release the video capture
    cap.release()

    return captions

def process_frame(cap, i, model, transform, total_frames):
    # Set the video capture to the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    # Read the frame from the video capture
    ret, frame = cap.read()

    # Check if there are no more frames
    if not ret:
        return None

    # Convert the frame to RGB and apply the transformation
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im = transform(im).unsqueeze(0)

    # Run inference on the frame
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im)

    # Decode the output caption
    caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

    # Print the progress
    print(f"Processed frame {i+1} of {total_frames}")

    return caption

def write_output_to_file(filename, captions):
    output_dict = {"captions": captions}
    with open(filename, "w") as f:
        json.dump(output_dict, f)

def read_captions_from_file(filename):
    with open(filename, "r") as f:
        output_dict = json.load(f)
        return output_dict["captions"]
    
    
