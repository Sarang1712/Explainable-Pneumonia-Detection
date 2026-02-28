import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. Custom Grad-CAM Implementation
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        model_output = self.model(input_tensor)
        self.model.zero_grad()
        target = model_output[:, class_idx]
        target.backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap

# ==========================================
# 2. Main Inference Engine
# ==========================================
def run_inference(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['NORMAL', 'PNEUMONIA']

    # Load Model
    print("Loading ResNet-18 architecture")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Processing image: {image_path}")
    original_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # Initialize Grad-CAM
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    prediction = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100
    print(f"\n[RESULT] Diagnosed as: {prediction} (Confidence: {conf_score:.2f}%)")

    # Generate Heatmap
    with torch.enable_grad():
        input_tensor.requires_grad = True
        heatmap = grad_cam.generate_heatmap(input_tensor, predicted_idx.item())

    # Overlay and Save
    img_cv = cv2.cvtColor(np.array(original_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap_cv = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_cv, 0.4, 0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original X-Ray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM: {prediction}\nConf: {conf_score:.1f}%")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("xai_inference_result.png", dpi=300)
    print("Saved Explainable AI visualization as 'xai_inference_result.png'")
    
    plt.show()

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    # ---> JUST CHANGE THESE TWO NAMES TO TEST YOUR FILES <---
    TEST_IMAGE_PATH = "sample_xray.jpeg" 
    MODEL_WEIGHTS_PATH = "pneumonia_final_best.pth"
    
    run_inference(TEST_IMAGE_PATH, MODEL_WEIGHTS_PATH)
