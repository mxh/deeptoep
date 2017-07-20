from network import *
from game import *
import codecs, json
import sys

def usage():
    print("Usage: [output_json_filename]")

if __name__=="__main__":
    if len(sys.argv) < 2:
        usage()
        exit()
    trainer = ToepQNetworkTrainer()

    wanted_variables = ["MainNet/FeatureExtraction/FeatureExtraction/Hidden1/weights:0",
                        "MainNet/FeatureExtraction/FeatureExtraction/Hidden1/biases:0",
                        "MainNet/FeatureExtraction/FeatureExtraction/Hidden2/weights:0",
                        "MainNet/FeatureExtraction/FeatureExtraction/Hidden2/biases:0",
                        "TargetNet/Value/fully_connected/weights:0",
                        "TargetNet/Value/fully_connected/biases:0",
                        "TargetNet/Advantage/fully_connected/weights:0",
                        "TargetNet/Advantage/fully_connected/biases:0"]

    all_tf_variables = tf.trainable_variables()
    wanted_tf_variables = [[]] * len(wanted_variables)
    for tf_variable in all_tf_variables:
        if tf_variable.name in wanted_variables:
            wanted_tf_variables[wanted_variables.index(tf_variable.name)] = tf_variable

    wanted_tf_values = trainer.session.run(wanted_tf_variables)

    json_out = {}
    json_out["hidden_1"] = {}
    json_out["hidden_1"]["weights"] = wanted_tf_values[0].tolist()
    json_out["hidden_1"]["biases"] = wanted_tf_values[1].tolist()
    json_out["hidden_2"] = {}
    json_out["hidden_2"]["weights"] = wanted_tf_values[2].tolist()
    json_out["hidden_2"]["biases"] = wanted_tf_values[3].tolist()
    json_out["value"] = {}
    json_out["value"]["weights"] = wanted_tf_values[4].tolist()
    json_out["value"]["biases"] = wanted_tf_values[5].tolist()
    json_out["advantage"] = {}
    json_out["advantage"]["weights"] = wanted_tf_values[6].tolist()
    json_out["advantage"]["biases"] = wanted_tf_values[7].tolist()

    json.dump(json_out, codecs.open(sys.argv[1], 'w', encoding='utf-8'), separators=(", ", ": "), sort_keys=True, indent=4)
