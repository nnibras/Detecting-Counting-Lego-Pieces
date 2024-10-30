# Import necessary libraries
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
from LegoDataset import LegoDataset
from collections import defaultdict

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
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # Set to evaluation mode

# Set up transformations
transform = T.Compose([T.ToTensor()])

# Load validation dataset
validation_dir = "val"
ann_file_val = "lego_coco_annotations_val.json"  # Validation annotation JSON
validation_dataset = LegoDataset(
    root=validation_dir, annFile=ann_file_val, transforms=transform
)
validation_loader = DataLoader(
    validation_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
)


# Function to calculate mAP, Precision, and Recall
def evaluate_model(data_loader, model, iou_thresholds=[0.5, 0.75]):
    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output["boxes"]
                gt_boxes = targets[i]["boxes"].to(device)

                # Compute IoUs
                iou_matrix = box_iou(pred_boxes, gt_boxes)

                for iou_thresh in iou_thresholds:
                    tp = (iou_matrix.max(dim=1)[0] >= iou_thresh).sum().item()
                    fp = len(pred_boxes) - tp
                    fn = len(gt_boxes) - tp

                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    results[f"mAP@{iou_thresh}"].append(precision)
                    results[f"Recall@{iou_thresh}"].append(recall)

    # Calculate mean values for each metric
    for metric in results:
        avg_value = sum(results[metric]) / len(results[metric])
        print(f"{metric}: {avg_value:.4f}")


evaluate_model(validation_loader, model)
print("Evaluation with mAP calculation complete.")
