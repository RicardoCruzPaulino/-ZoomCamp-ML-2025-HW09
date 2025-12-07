
import urllib.request

def download_files():
    prefix = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
    data_url = f"{prefix}/hair_classifier_v1.onnx.data"
    model_url = f"{prefix}/hair_classifier_v1.onnx"

    # Download data file
    print(f"Downloading {data_url}...")
    urllib.request.urlretrieve(data_url, "hair_classifier_v1.onnx.data")
    print("Saved hair_classifier_v1.onnx.data")

    # Download model file
    print(f"Downloading {model_url}...")
    urllib.request.urlretrieve(model_url, "hair_classifier_v1.onnx")
    print("Saved hair_classifier_v1.onnx")

if __name__ == "__main__":
    download_files()