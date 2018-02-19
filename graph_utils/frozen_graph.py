"""Class that encapsulates things related to frozen graphs.
"""

import tensorflow as tf

from freeze_graph import freeze_graph_with_def_protos


class FrozenGraph:

  def __init__(self, frozen_graph_def, input_names, output_names):
    self._frozen_graph_def = frozen_graph_def
    self._input_names = input_names
    self._output_names = output_names

  def output_as_saved_model(self, export_path):
    # Load graph
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(self._frozen_graph_def, name="")
      # TODO(lamberce): generalize to work with more than 1 input/output
      input_tensor = graph.get_tensor_by_name(self._input_names[0])
      output_tensor = graph.get_tensor_by_name(self._output_names[0])

    prediction_signature = (
      tf.saved_model.signature_def_utils.predict_signature_def(
          inputs={"inputs": input_tensor},
          outputs={"outputs": output_tensor}))

    sig_def_map = {
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature,
    }

    saved_model_builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # create session as this is needed by SavedModelBuilder
    with tf.Session(graph=graph) as sess:
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map=sig_def_map)
    saved_model_builder.save()

  @staticmethod
  def load_from_file(filepath, input_names, output_names):
    with tf.gfile.GFile(filepath, "rb") as graph_file:
      frozen_graph_def = tf.GraphDef()
      frozen_graph_def.ParseFromString(graph_file.read())

    return FrozenGraph(frozen_graph_def, input_names, output_names)
