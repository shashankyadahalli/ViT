import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the saved model from disk
model_path = 'D:/VisionTransformer/best_model.pt'
model = torch.load(model_path)

# Prepare the input data in the appropriate format
input_image = Image.open('D:/VisionTransformer/Train/0a583b7fbfb10dc55cc04dfb4ca8e39b26937ea3.tif')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = transform(input_image).unsqueeze(0)

# Pass the input data to the model to generate predictions
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# Process the output of the model to generate the final result
predicted_class = torch.argmax(output).item()


print(f'The predicted label is: {predicted_class}')
