import onnxruntime as rt

# Load the model
sess = rt.InferenceSession("hair_classifier_v1.onnx", providers=["CPUExecutionProvider"])

# Inspect inputs
for i, inp in enumerate(sess.get_inputs()):
    print(f"Input {i}: name={inp.name}, shape={inp.shape}, type={inp.type}")

# Inspect outputs
for i, out in enumerate(sess.get_outputs()):
    print(f"Output {i}: name={out.name}, shape={out.shape}, type={out.type}")