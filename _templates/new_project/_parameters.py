# Imports
# =============================================================================
import os
from distutils.util import strtobool
from argparse import ArgumentParser, Namespace

# Parameter Configuration
# =============================================================================
# Argument Parser
main_arg_parser = ArgumentParser(description="parser for fast-neural-style")

# Main Parameters
main_arg_parser.add_argument("--main_debug", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--main_eval", type=strtobool, default=True, help="")
main_arg_parser.add_argument("--main_seed", type=int, default=69, help="")

# Project
main_arg_parser.add_argument("--project_name", type=str, default='traj-gen', help="")
main_arg_parser.add_argument("--project_owner", type=str, default='si11ium', help="")
main_arg_parser.add_argument("--project_neptune_key", type=str, default=os.getenv('NEPTUNE_KEY'), help="")

# Data Parameters
main_arg_parser.add_argument("--data_worker", type=int, default=10, help="")
main_arg_parser.add_argument("--data_dataset_length", type=int, default=10000, help="")
main_arg_parser.add_argument("--data_root", type=str, default='data', help="")
main_arg_parser.add_argument("--data_additional_resource_root", type=str, default='res/resource/root', help="")
main_arg_parser.add_argument("--data_use_preprocessed", type=strtobool, default=True, help="")

# Transformations
main_arg_parser.add_argument("--transformations_to_tensor", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--transformations_normalize", type=strtobool, default=False, help="")

# Transformations
main_arg_parser.add_argument("--train_outpath", type=str, default="output", help="")
main_arg_parser.add_argument("--train_version", type=strtobool, required=False, help="")
main_arg_parser.add_argument("--train_epochs", type=int, default=500, help="")
main_arg_parser.add_argument("--train_batch_size", type=int, default=200, help="")
main_arg_parser.add_argument("--train_lr", type=float, default=1e-3, help="")
main_arg_parser.add_argument("--train_num_sanity_val_steps", type=int, default=0, help="")

# Model
main_arg_parser.add_argument("--model_type", type=str, default="CNNRouteGenerator", help="")

main_arg_parser.add_argument("--model_activation", type=str, default="leaky_relu", help="")
main_arg_parser.add_argument("--model_use_bias", type=strtobool, default=True, help="")
main_arg_parser.add_argument("--model_use_norm", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--model_dropout", type=float, default=0.00, help="")

# Model 2: Layer Specific Stuff
main_arg_parser.add_argument("--model_filters", type=str, default="[16, 32, 64]", help="")
main_arg_parser.add_argument("--model_features", type=int, default=16, help="")

# Parse it
args: Namespace = main_arg_parser.parse_args()

if __name__ == '__main__':
    pass