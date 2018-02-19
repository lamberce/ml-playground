"""A client that talks to tensorflow_model_server to perform inference."""

import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

class TensorflowModelEvaluator:
  """Handles making calls to tensorflow_model_server for tensorflow models."""

  def __init__(self, hostport, model_name, timeout):
    # TODO(lamberce): look into if this needs to be secure
    channel = grpc.insecure_channel(hostport)
    self._stub = prediction_service_pb2.PredictionServiceStub(channel)
    self._model_name = model_name
    self._timeout = timeout

  def perform_inference(self, input):
    """Calls tensorflow_model_server to perform inference on given input.

    Args:
      input: numpy array that we are going to perform inference on.

    Returns:
      The result of performing inference.
    """
    prediction_request = self._build_prediction_request(input)
    # TODO(lamberce): look into making this non-blocking (stub.Predict.future)
    prediction_result = self._stub.Predict(prediction_request, self._timeout)
    return numpy.array(prediction_result.outputs['outputs'].float_val)

  def _build_prediction_request(self, input):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = (
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(input, shape=input.shape))
    return request
