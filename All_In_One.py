import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools.coco import COCO
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
from collections import defaultdict

# Set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Step 1: Convert VOC Annotations to COCO Format
def voc_to_coco(data_dir, output_file):
    categories = [{"id": 1, "name": "lego"}]
    images, annotations = [], []
    ann_id = 1

    for i, filename in enumerate(os.listdir(data_dir)):
        if filename.endswith(".xml"):
            xml_path = os.path.join(data_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_name = root.find("filename").text
            img_path = os.path.join(data_dir, img_name)
            img = Image.open(img_path)
            width, height = img.size

            image_info = {
                "id": i + 1,
                "file_name": img_name,
                "width": width,
                "height": height,
            }
            images.append(image_info)

            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                x_min = int(bbox.find("xmin").text)
                y_min = int(bbox.find("ymin").text)
                x_max = int(bbox.find("xmax").text)
                y_max = int(bbox.find("ymax").text)
                w = x_max - x_min
                h = y_max - y_min

                ann = {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": 1,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
                annotations.append(ann)
                ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_file, "w") as f:
        json.dump(coco_data, f)


print("Step 1: Conversion to COCO format started.")
voc_to_coco("test", "lego_coco_annotations.json")
print("Step 1: Conversion to COCO format completed.")


# Step 2: Create the LegoDataset Class
class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []  # Dummy masks

        for ann in annotations:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(1)  #'lego' is the only class, labeled as 1

            # Dummy mask for Mask R-CNN, filled with zeros
            dummy_mask = np.zeros(
                (img_info["height"], img_info["width"]), dtype=np.uint8
            )
            masks.append(dummy_mask)

        # Convert to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(
            np.array(masks), dtype=torch.uint8
        )  # Dummy masks as tensor

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,  # Dummy masks added
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


# Set up transformations
transform = T.Compose([T.ToTensor()])
print("Step 2: Initializing dataset and dataloader.")

# Initialize dataset and data loader
dataset = LegoDataset(
    root="test", annFile="lego_coco_annotations.json", transforms=transform
)
data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)

# Step 3: Define and load model with correct predictors
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes=2
)
model.to(device)

print("Step 3: Model initialization complete.")

# Step 4: Set up the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Step 5: Training Loop
print("Model Training Started!")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in data_loader:
        # Move each image and target to the device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Checkpoint to confirm images and targets are on the correct device
        print("Images moved to device:", [img.device for img in images])
        print(
            "Targets moved to device:",
            [{k: v.device for k, v in t.items()} for t in targets],
        )

        # Compute losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Step 5: Training complete.")

# Save the trained model
model_save_path = "mask_rcnn_lego.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# Step 6: Model Evaluation (mAP Calculation with IoU 0.5)
def calculate_mAP(data_loader, model, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        iou_values = []
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            # Compare predicted boxes with ground truth
            for i, output in enumerate(outputs):
                pred_boxes = output["boxes"]
                gt_boxes = targets[i]["boxes"].to(device)

                iou_matrix = box_iou(pred_boxes, gt_boxes)
                iou_values.extend(
                    iou_matrix.max(dim=1)[0].cpu().tolist()
                )  # Max IoU per prediction

        # Calculate mAP for IoU threshold
        true_positives = sum(iou >= iou_threshold for iou in iou_values)
        mAP = true_positives / len(iou_values) if iou_values else 0
        print(f"mAP at IoU {iou_threshold}: {mAP:.4f}")


print("Step 6: Evaluating model with mAP at IoU 0.5.")
calculate_mAP(data_loader, model)


# Visualization Check: Display an example with predictions
def visualize_predictions(images, outputs):
    for i in range(len(images)):
        img = images[i].cpu()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(F.to_pil_image(img))

        boxes = outputs[i]["boxes"].cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
            )
            ax.add_patch(rect)

        plt.show()
        break


print("Step 7: Visualizing sample predictions.")
with torch.no_grad():
    images, _ = next(iter(data_loader))
    images = [img.to(device) for img in images]
    outputs = model(images)
    visualize_predictions(images, outputs)

print("Training, evaluation, and visualization completed.")
