FROM public.ecr.aws/lambda/python:3.13

# Install required dependencies
RUN pip install --no-cache-dir pillow numpy onnxruntime

# Copy ONNX model files
COPY worshop-serveless/hair_classifier_v1.onnx ${LAMBDA_TASK_ROOT}/
COPY worshop-serveless/hair_classifier_v1.onnx.data ${LAMBDA_TASK_ROOT}/

# Copy Lambda handler
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to the Lambda handler
CMD [ "lambda_function.lambda_handler" ]
