import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
from io import BytesIO
import base64
import pdb

with open("preprocessor_config.json") as f:
    pre = json.loads(f.read())
    
with open("config.json") as f:
    id2label = json.loads(f.read())["id2label"]

def preprocess_image(base64_data):
    input_size = (pre["size"]["height"], pre["size"]["width"])
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pre["image_mean"], std=pre["image_std"]),
        ]
    )
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

def data_augmentation(base64_data):
    
    num_images = 10
    images = [] 
    input_size = (pre["size"]["height"], pre["size"]["width"])
    input_image = Image.open(BytesIO(base64.b64decode(base64_data)))
    
    for i in range(num_images):
    
        transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
            ]
        )
        image = transform(input_image).unsqueeze(0)
        image = image.numpy()
        images.append(image)
    
    return images


def run_inference(base64_data):
    
    model_path = "./inference_artifacts/inference.onnx"
    session = ort.InferenceSession(model_path)  
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    image = preprocess_image(base64_data)
    results = session.run([output_name], {input_name: image})
    

    return results[0]

def softmax(z):
    probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return probabilities

def generate_target_logits(target_class_index, num_classes, large_value, low_value):
    logits = np.full((1, num_classes), low_value, dtype=np.float32)
    logits[0][target_class_index] = large_value
    
    return logits

def get_labels(base64_data):
    
    output = run_inference(base64_data)
    probabilities = softmax(output)[0]
    
    sorted_indices = np.argsort(-probabilities)
    
    labels = {}
    for x, i in enumerate(sorted_indices):
        if x > 4:
            break
        labels[id2label[str(i)]] = str(probabilities[i])
    
    return labels