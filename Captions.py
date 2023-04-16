import open_clip
import torch
from PIL import Image
import cv2
import json

def WatchVideo(video_path):
    # Load the model
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )



    # Define the video path
    video_path = video_path

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an empty list to store the output captions
    captions = []

    # Loop through every 5th frame of the video
    for i in range(0, total_frames, 5):
        # Set the video capture to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame from the video capture
        ret, frame = cap.read()
        
        # Check if there are no more frames
        if not ret:
            break
        
        # Convert the frame to RGB and apply the transformation
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im = transform(im).unsqueeze(0)
        
        # Run inference on the frame
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.generate(im)
        
        # Decode the output caption
        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        
        # Append the caption to the list
        captions.append(caption)
        
        # Print the progress
        print(f"Processed frame {i+1} of {total_frames}")
        
    # Release the video capture
    cap.release()

    # Create a dictionary to store the captions
    output_dict = {"captions": captions}

    # Write the output to a JSON file
    with open("output.json", "w") as f:
        json.dump(output_dict, f)
