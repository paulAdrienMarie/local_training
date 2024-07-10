from aiohttp import web
import platform
assert platform.python_version_tuple() == ('3', '9', '6')
from transformers import pipeline 
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from utils import get_labels, preprocess_image, generate_target_logits
import onnx
import pdb
import json
import numpy as np

# path to training artifacts
path_to_training = "./training_artifacts/training_model.onnx"
path_to_eval = "./training_artifacts/eval_model.onnx"
path_to_optimizer = "./training_artifacts/optimizer_model.onnx"
path_to_checkpoint = "./training_artifacts/checkpoint"

with open("config.json") as f:
    label2id = json.loads(f.read())["label2id"]

async def index(request):
	return web.FileResponse("./index.html")

async def js_handler(request):
	return web.FileResponse("./script.js")

async def css_handler(request):
    return web.FileResponse("./style.css")

async def classifier_handler(request):
    
    try:
       
        data = await request.json()
        input_image = data.get('input_image','')
        
        if ',' in input_image:
             _ , base64_data = input_image.split(',', 1)
        else:
             base64_data = input_image
        
        labels = get_labels(base64_data)
        # print(f"Class {i} ~ {id2label[str(i)]} ~ {probabilities[i]:.4f}")
        
        
        return web.json_response({"object_labels": labels })
    except Exception as e:
        return web.json_response({ 'error': str(e) })
    

async def training_handler(request):
    
    try:
        data = await request.json()
        input_image = data.get('input_image','')
        correct_class = data.get('correct_class','')
        
        if ',' in input_image:
            _, base64_data = input_image.split(',',1)
        else:
            base64_data = input_image
            
        state = CheckpointState.load_checkpoint(path_to_checkpoint)
        
        module = Module(
            path_to_training,
            state,
            path_to_eval,
            device="cpu"
        )
                
        optimizer = Optimizer(path_to_optimizer,module)
        pdb.set_trace()
        optimizer.get_learning_rate()
        
        num_epoch = 3
        pixel_values = preprocess_image(base64_data)
        target_probs = np.zeros(shape=(1,1000),dtype=np.float32)
        
        index = label2id[str(correct_class)]
        
        target_output = generate_target_logits(target_class_index=index,num_classes=1000,large_value=10)
                
        for epoch in range(num_epoch):
            # set the module in training mode
            module.train()
            loss=0
            # update the loss 
            loss += module(pixel_values,target_output)
            # start gradient descent
            optimizer.step()
            # reset the gradient for the next training phase
            module.lazy_reset_grad()
            # evaluation of the model∏
            module.eval()
            print(f"Epoch {epoch} - Training loss: {loss}")
            
        CheckpointState.save_checkpoint(state, path_to_checkpoint)
                
        module.export_model_for_inferencing("./inference_artifacts/inference.onnx",["logits"])
        print("Model has been updated")
        
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    else:
        return web.json_response({"message": "Model has been trained successfully on {} epochs".format(num_epoch) })
    
    

