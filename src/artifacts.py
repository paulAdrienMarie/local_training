import platform

assert platform.python_version_tuple() == ('3', '9', '6')

from onnxruntime.training import artifacts
import onnx
from onnx.checker import check_model
import pdb


# Load the ONNX model
onnx_model = onnx.load("./inference_artifacts/initial_model.onnx")

# Check if the model has been loaded successfully
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: {}".format(e))
else:
    print("The exported model is valid!")

model_initializers = onnx_model.graph.initializer
requires_grad = []
frozen_params = []

for initializer in model_initializers:
    requires_grad.append(initializer.name)

path_to_output_artifacts = "./training_artifacts"

artifacts.generate_artifacts(
    onnx_model,
    optimizer=artifacts.OptimType.AdamW,
    loss=artifacts.LossType.L1Loss,
    artifact_directory=path_to_output_artifacts,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
)

