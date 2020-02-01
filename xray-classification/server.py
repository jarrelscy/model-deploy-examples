import logging
from io import BytesIO
import cv2
import msgpack
import pydicom
import numpy as np
import torch
from model.classifier import Classifier
from easydict import EasyDict as edict
import torch.nn.functional as F
from skimage.exposure import equalize_adapthist
from flask import Flask, Response, abort, request
from PIL import Image
import json
from waitress import serve

app = Flask(__name__)
model = None

config = {"input_shape": (3, 512, 512)}
threshs = np.array([[ -8.159552],
       [ -5.743932],
       [ -8.048886],
       [-11.211817],
       [ -5.080043],
       [ -9.686677]], dtype=float)

def load_model():
    global model
    with open('config/example.json') as f:
        cfg = edict(json.load(f))
    jf_model = Classifier(cfg)
    jf_model.cfg.num_classes = [1,1,1,1,1,1]
    jf_model._init_classifier()    
    jf_model._init_attention_map()
    jf_model._init_bn()
    jf_model.load_state_dict(torch.load('model_best.pt'))
    model = jf_model.cuda()


def prepare_input(im):
    # convert grayscale to RGB
    assert len(im.shape) == 2
    
    im = cv2.resize(im, (1024, 1024))
    im = equalize_adapthist(im.astype(float) / im.max(), clip_limit=0.01)    
    im = cv2.resize(im, (512, 512))    
    im = im * 2 - 1
    im = np.array([[im, im, im]])
    return torch.from_numpy(im).cuda()


def predict(x):
    with torch.no_grad():
        y_prob = model(x)
        y_prob = torch.cat(y_prob[0], dim=1).detach().cpu()
        y_prob = F.sigmoid(y_prob + torch.from_numpy(threshs).reshape((1,6)))
        y_classes = (y_prob > 0.5).numpy().astype(int)

        class_index = y_classes
        probability = y_prob.numpy()


        result = {
            "class_index": list(y_classes),
            "data": None,
            "probability": list(probability),
            "explanations": [           
            ],
        }
    return result


@app.route("/inference", methods=["POST"])
def inference():
    """
    Route for model inference.

    The POST body is msgpack-serialized binary data with the follow schema:

    {
        "instances": [
            {
                "file": "bytes"
                "tags": {
                    "StudyInstanceUID": "str",
                    "SeriesInstanceUID": "str",
                    "SOPInstanceUID": "str",
                    ...
                }
            },
            ...
        ],
        "args": {
            "arg1": "str",
            "arg2": "str",
            ...
        }
    }

    The `file bytes is the raw binary data representing a DICOM file, and can be loaded using
    `pydicom.dcmread()`.

    The response body should be the msgpack-serialized binary data of the results:

    [
        {
            "study_uid": "str",
            "series_uid": "str",
            "instance_uid": "str",
            "frame_number": "int",
            "class_index": "int",
            "data": {},
            "probability": "float",
            "explanations": [
                {
                    "name": "str",
                    "description": "str",
                    "content": "bytes",
                    "content_type": "str",
                },
                ...
            ],
        },
        ...
    ]
    """
    if not request.content_type == "application/msgpack":
        abort(400)

    data = msgpack.unpackb(request.get_data(), raw=False)
    input_instances = data["instances"]
    input_args = data["args"]

    results = []

    for instance in input_instances:
        try:
            tags = instance["tags"]
            ds = pydicom.dcmread(BytesIO(instance["file"]))
            image = ds.pixel_array
            result = predict(prepare_input(image))
            result["study_uid"] = tags["StudyInstanceUID"]
            result["series_uid"] = tags["SeriesInstanceUID"]
            result["instance_uid"] = tags["SOPInstanceUID"]
            result["frame_number"] = None
            results.append(result)
        except Exception as e:
            logging.exception(e)
            abort(500)

    resp = Response(msgpack.packb(results, use_bin_type=True))
    resp.headers["Content-Type"] = "application/msgpack"
    return resp


@app.route("/healthz", methods=["GET"])
def healthz():
    return "", 200


if __name__ == "__main__":
    load_model()
    serve(app, listen="*:6324")
