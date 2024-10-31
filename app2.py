import torch
import gradio as gr
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load and configure the Mask R-CNN model with 2 classes
model_path = "mask_rcnn_lego.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "The model file 'mask_rcnn_lego.pth' was not found in the directory."
    )

model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Update the box predictor head to match 2 classes (background + LEGO)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Update the mask predictor head to match 2 classes
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes=2
)

# Now, load the state_dict for your custom model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Set up transformations
transform = T.Compose([T.ToTensor()])


# Function to create pseudo-masks based on bounding boxes
def create_pseudo_mask(image, box):
    mask = Image.new("L", image.size, 0)  # Create a blank mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)  # Fill in the bounding box area
    return mask


# Function to process image with pseudo-mask visualization and bounding boxes
def detect_legos(image, use_pseudo_masks=True):
    # Apply transformations
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions with the custom model
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract boxes and scores above threshold
    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    thresholded_indices = [i for i, score in enumerate(scores) if score >= 0.5]
    boxes = boxes[thresholded_indices]
    num_legos_detected = len(boxes)

    # Draw pseudo-masks on the image first
    image_with_masks = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box

        # Use pseudo-masks based on bounding boxes
        if use_pseudo_masks:
            mask_img = create_pseudo_mask(image, [x1, y1, x2, y2])
            mask_img = ImageOps.colorize(
                mask_img.convert("L"), black="blue", white="blue"
            ).convert("RGBA")
            image_with_masks.paste(mask_img, (0, 0), mask_img)

    # Draw the bounding boxes on top of the masks for better visibility
    draw = ImageDraw.Draw(image_with_masks)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle(
            [x1, y1, x2, y2], outline="yellow", width=3
        )  # Draw yellow bounding box

    # Set title with count of detected LEGO pieces
    title = f"Detected LEGO pieces: {num_legos_detected}"

    return image_with_masks, title


# Gradio interface function
def gradio_interface(image):
    image_with_masks, title = detect_legos(image, use_pseudo_masks=True)
    return image_with_masks, title


# Set up Gradio Interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox(label="Detection Summary")],
    title="LEGO Detection with Mask R-CNN",
    description="Upload an image to detect and count LEGO pieces with bounding boxes and simulated masks.",
)

# Launch the Gradio app
interface.launch()
