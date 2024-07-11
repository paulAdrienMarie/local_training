# Web app for local training in Python/JS

This application aims to show a little web application for local training using onnxruntime training api.

## Dependencies: 

Create a venv at the root of the project, run : `python3 -m venv env` and activate it by running : `source env/bin/activate`.

Make sure to run with `python 3.9.6`

Run : `pip install -r requirements.txt`

Install `onnx:runtimetraining:api` by running :

`python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu`


## Launch the app 

Run `python watcher.py`

