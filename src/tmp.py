import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
from io import BytesIO
import base64
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
import math
import pdb

# Paths to the model files
model_path = "./training_artifacts/true_model.onnx"
decoder_path = "./training_artifacts/decoder_model.onnx"

# Load ONNX models
encoder_session = ort.InferenceSession(model_path)
decoder_session = ort.InferenceSession(decoder_path)

# Define a custom Linear layer class
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

# Function to preprocess the image
def preprocess_image(base64_data):
    input_size = (224, 224)
    image = Image.open(BytesIO(base64.b64decode(base64_data)))

    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    image_normalized = transform(image).unsqueeze(0)  # Add batch dimension
    pdb.set_trace()
    return image_normalized.detach().numpy()

# Function to run inference using the encoder model
def run_inference(image, linear_layer):
    input_name = encoder_session.get_inputs()[0].name
    output_name = encoder_session.get_outputs()[0].name

    last_hidden_state = encoder_session.run([output_name], {input_name:image})
    pdb.set_trace()
    last_hidden_state = torch.tensor(last_hidden_state[0],dtype=torch.float32)  # Convert to PyTorch tensor
    
    # Use only the CLS token's hidden state
    logits = linear_layer(last_hidden_state[:, 0, :])
    return logits.detach().numpy()  # Convert back to NumPy array

# Function to postprocess the output logits
def postprocess_output(logits):
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    return probabilities

# Function to get labels from the config file
def get_labels():
    with open("config.json") as f:
        data = json.load(f)["id2label"]
    return data

# Function to get object labels from the image
def get_objects_labels(image, in_features, out_features):
    linear_layer = Linear(in_features, out_features)
    logits = run_inference(image, linear_layer)
    pdb.set_trace()
    probabilities = postprocess_output(logits)
    pdb.set_trace()
    sorted_indices = np.argmax(probabilities)  # Assuming batch size 1
    labels = get_labels()
    return labels[str(sorted_indices)], logits

# Function to run the decoder
def run_decoder(input_ids, last_hidden_states):
    output_name = decoder_session.get_outputs()[0].name
    results = decoder_session.run([output_name], {'input_ids': input_ids, 'last_hidden_states': last_hidden_states})
    return results

# Example usage:
# Assuming `base64_data` contains the base64 encoded image data
# base64_data = "..."

# Preprocess the image
# image = preprocess_image(base64_data)

# Get the object labels
# in_features = 768  # Example input feature size, should match your encoder model's output size
# out_features = 1000  # Example output feature size, should match the number of classes
# labels, logits = get_objects_labels(image, in_features, out_features)
# print(labels)
