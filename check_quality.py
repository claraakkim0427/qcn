import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import pandas as pd

image_folder = 'dataset'  
weight_file = 'pretrained/spaq.pth'  

class SimpleResNetWrapper(torch.nn.Module):
    def __init__(self):
        # uses ResNet-50, but removes the final classification layer
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])  
        # adds a new fully connected layer to output a single quality score
        self.fc = torch.nn.Linear(2048, 1)  

    def forward(self, x):
        # pass img thru encoder, flatten it, feed it thru fc 
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_model(weight_path):
    model = SimpleResNetWrapper()
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def preprocess_image(image_path):
    # image processing: resize, conver to a normalized tensor
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def assess_images(image_dir, weight_path, save_csv='quality_results.csv'):
    model = load_model(weight_path)
    print(f"\nScanning folder: {image_dir}\n")

    results = []

    for root, _, files in os.walk(image_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(root, fname)
            try:
                img_tensor = preprocess_image(fpath)
                with torch.no_grad():
                    score_tensor = model(img_tensor)
                    score = score_tensor.item()
                label = 'Good' if score >= 5 else 'Poor'
                print(f"Processing: {fpath}\n   → Quality Score: {score:.4f} → {label}\n")
                results.append({
                    'file': fpath, 'score': round(score, 4), 'label': label
                })
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
        
        df = pd.DataFrame(results)
        df.to_csv(save_csv, index=False)

if __name__ == "__main__":
    assess_images(image_folder, weight_file)
