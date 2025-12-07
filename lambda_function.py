import json
import sys
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
import numpy as np
import onnxruntime as rt


def preprocess_image(image_source, target_size=(200, 200)):
    """Load image from URL or file path, preprocess it."""
    if isinstance(image_source, str) and (image_source.startswith("http://") or image_source.startswith("https://")):
        # Download from URL
        with urlopen(image_source) as response:
            img = Image.open(BytesIO(response.read())).convert("RGB")
    else:
        # Load from file path
        img = Image.open(image_source).convert("RGB")
    
    if img.size != target_size:
        img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr_chw = arr.transpose(2, 0, 1)
    # R channel first pixel 
    print('arr_chw.shape', arr_chw.shape)
    print('R channel first pixel (arr_chw[0,0,0]):', arr_chw[0,0,0])

    return np.expand_dims(arr_chw, axis=0)


def infer(image_source, model_path="hair_classifier_v1.onnx"):
    """Run inference on an image."""
    input_arr = preprocess_image(image_source)
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_arr})
    raw = float(np.array(outputs[0])[0][0])
    prob = float(1.0 / (1.0 + np.exp(-raw)))
    return raw, prob


def lambda_handler(event, context):
    """AWS Lambda handler function.
    
    Expected event:
    {
        "image_url": "https://example.com/image.jpg"  # or local path
    }
    """
    try:
        image_source = event.get("image_url") or event.get("image")
        if not image_source:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'image_url' or 'image' in request"})
            }
        
        raw, prob = infer(image_source)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "raw_output": raw,
                "sigmoid_probability": prob
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
