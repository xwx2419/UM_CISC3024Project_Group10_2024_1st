import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import numpy as np

# Define the VGG model
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the model
model_path = "checkpoints/best.pth"
model = VGG(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')

    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL Image or numpy array, got something else.")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Prediction function
def predict_digit(image):
    image = preprocess(image)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probs = probabilities.squeeze().tolist()
    predicted_class = probs.index(max(probs))
    return {str(i): prob for i, prob in enumerate(probs)}, predicted_class

# Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", image_mode="L"),  # Correct usage in Gradio 3.x
    outputs=["json", "label"],
    live=True,
    title="Digit Recognition",
    description="Draw a digit (0-9) on the sketchpad and the model will classify it."
)

# Launch the Gradio app
iface.launch()
