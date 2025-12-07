#!/usr/bin/env python3
"""Simple inference script for `hair_classifier_v1.onnx`.

Downloads an image from a URL (or loads from disk), resizes to (200,200), 
normalizes to [0,1], reorders to CHW, runs the ONNX model and prints the 
raw output and sigmoid probability.
"""
import argparse
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
import numpy as np
import onnxruntime as rt


def preprocess_image(image_source, target_size=(200, 200)):
    """Load image from URL or file path, preprocess it.
    
    Args:
        image_source: URL (str starting with http) or file path
        target_size: tuple (width, height)
    
    Returns:
        Preprocessed image as (1, 3, 200, 200) numpy array
    """
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


def infer(image_source, model_path):
    """Run inference on an image (from URL or file path).
    
    Args:
        image_source: URL or file path to image
        model_path: path to ONNX model
    
    Returns:
        (raw_output, sigmoid_probability)
    """
    input_arr = preprocess_image(image_source)
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_arr})
    raw = np.array(outputs[0])
    prob = 1.0 / (1.0 + np.exp(-raw))
    return raw, prob


def main():
    parser = argparse.ArgumentParser(description="Run inference on an image from URL or file path with hair_classifier_v1.onnx")
    parser.add_argument("image", help="URL (http/https) or file path to image")
    parser.add_argument("--model", "-m", default="hair_classifier_v1.onnx", help="Path to ONNX model (default: hair_classifier_v1.onnx)")
    args = parser.parse_args()

    raw, prob = infer(args.image, args.model)
    print("raw output:", raw)
    print("sigmoid probability:", prob)


if __name__ == "__main__":
    main()
