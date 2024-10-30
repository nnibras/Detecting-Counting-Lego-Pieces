import torch
import gradio as gr
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the saved model
model_path = "mask_rcnn_lego.pth"
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes=2
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Set up transformations
transform = T.Compose([T.ToTensor()])


# Function to process image and return bounding boxes and count
def detect_legos(image):
    # Apply transformations
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract boxes and draw them
    boxes = outputs[0]["boxes"].cpu().numpy()
    num_legos_detected = len(boxes)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Set a title with the count of detected objects
    title = f"Detected LEGO pieces: {num_legos_detected}"

    return image, title


# Gradio interface function
def gradio_interface(image):
    image_with_boxes, title = detect_legos(image)
    return image_with_boxes, title


# Set up Gradio Interface with the new API
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox(label="Detection Summary")],
    title="LEGO Detection with Mask R-CNN",
    description="Upload an image to detect and count LEGO pieces with bounding boxes.",
)

# Launch the Gradio app
interface.launch()
