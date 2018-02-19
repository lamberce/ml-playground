import argparse
import tensorflow as tf
from frozen_graph import FrozenGraph


FLAGS = None


def main():
  input_nodes = FLAGS.input_names.replace(" ", "").split(",")
  output_nodes = FLAGS.output_names.replace(" ", "").split(",")
  frozen_graph = FrozenGraph.load_from_file(FLAGS.frozen_model_filepath,
                                            input_nodes,
                                            output_nodes)
  frozen_graph.output_as_saved_model(FLAGS.saved_graph_export_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--frozen_model_filepath", type=str, default="",
                      help="Path to frozen model file to import.")
  parser.add_argument("--input_names", type=str, default="",
                      help="The names of the input nodes, comma separated.")
  parser.add_argument("--output_names", type=str, default="",
                      help="The names of the output nodes, comma separated.")
  parser.add_argument("--saved_graph_export_path", type=str, default="",
                      help="Path to output SavedModel.")
  FLAGS = parser.parse_args()
  main()