import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from PIL import Image
from torchvision import transforms
from network.network_utils import build_model
from configs.config_v1 import ConfigV1
from utils.train_utils import Maxlloyd
import numpy as np

def load_model(weight_path):
    cfg = ConfigV1()
    train_scores = np.arange(101)
    maxlloyd = Maxlloyd(train_scores, rpt_num=cfg.spv_num)
    cfg.score_pivot_score = maxlloyd.get_new_rpt_scores()
    cfg.reference_point_num = len(cfg.score_pivot_score)
    model = build_model(cfg)
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, cfg

def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def assess_images(folder_path, weight_path, threshold=0.5):
    model, cfg = load_model(weight_path)
    results = {}

    print(f"Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, fname)
                print(f"Processing: {path}")

                img_tensor = preprocess_image(path, cfg.image_size)

                with torch.no_grad():
                    # Call model the way it expects
                    score_tensor = model('extraction', {'img': img_tensor})
                    score = score_tensor.mean().item()

                results[path] = score >= threshold
                print(f"{path}: {'Good' if score >= threshold else 'Poor'} quality (score: {score:.2f})")

    return results

if __name__ == "__main__":
    image_folder = "dataset/"  
    weight_file = "pretrained/spaq.pth"
    assess_images(image_folder, weight_file)
