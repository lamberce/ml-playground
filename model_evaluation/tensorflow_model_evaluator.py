"""A client that talks to tensorflow_model_server to perform inference."""

import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# TODO(lamberce): put this in a common place to be used everywhere
# SavedModel input/ouput mapping names
_INPUT_NAME = "inputs"
_OUTPUT_NAME = "outputs"

class TensorflowModelEvaluator:
  """Handles making calls to tensorflow_model_server for tensorflow models."""

  def __init__(self, hostport, model_name, timeout):
    # TODO(lamberce): look into if this needs to be secure
    channel = grpc.insecure_channel(hostport)
    self._stub = prediction_service_pb2.PredictionServiceStub(channel)
    self._model_name = model_name
    self._timeout = timeout

  def perform_inference(self, inputs):
    """Calls tensorflow_model_server to perform inference on given input.

    Args:
      input: numpy array that we are going to perform inference on.

    Returns:
      The result of performing inference.
    """
    prediction_request = self._build_prediction_request(inputs)
    prediction_result = self._stub.Predict(prediction_request, self._timeout)
    return np.array(prediction_result.outputs[_OUTPUT_NAME].float_val)

  def perform_inference_async(self, inputs):
    """Calls tensorflow_model_server in non-blocking way to perform inference
    on given input.

    Args:
      input: numpy array that we are going to perform inference on.

    Returns:
      Future with callback attached that converts result to numpy array once
      response is recieved.
    """
    prediction_request = self._build_prediction_request(inputs)
    # TODO(lamberce): figure out how to add callback that transforms to numpy
    return self._stub.Predict.future(prediction_request, self._timeout)

  def _build_prediction_request(self, inputs):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = (
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    request.inputs[_INPUT_NAME].CopyFrom(
        tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))
    return request
