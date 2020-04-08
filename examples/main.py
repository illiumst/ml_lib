# Imports
# =============================================================================
import os
from distutils.util import strtobool
from pathlib import Path
from argparse import ArgumentParser, Namespace

import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ml_lib.modules.utils import LightningBaseModule
from ml_lib.utils.config import Config
from ml_lib.utils.logging import Logger
from ml_lib.utils.model_io import SavedLightningModels

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

_ROOT = Path(__file__).parent

# Parameter Configuration
# =============================================================================
# Argument Parser
main_arg_parser = ArgumentParser(description="parser for fast-neural-style")

# Main Parameters
main_arg_parser.add_argument("--main_debug", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--main_eval", type=strtobool, default=True, help="")
main_arg_parser.add_argument("--main_seed", type=int, default=69, help="")

# Data Parameters
main_arg_parser.add_argument("--data_worker", type=int, default=10, help="")
main_arg_parser.add_argument("--data_dataset_length", type=int, default=10000, help="")
main_arg_parser.add_argument("--data_root", type=str, default='data', help="")
main_arg_parser.add_argument("--data_map_root", type=str, default='res/shapes', help="")
main_arg_parser.add_argument("--data_normalized", type=strtobool, default=True, help="")
main_arg_parser.add_argument("--data_use_preprocessed", type=strtobool, default=True, help="")

main_arg_parser.add_argument("--data_mode", type=str, default='vae_no_label_in_map', help="")

# Transformations
main_arg_parser.add_argument("--transformations_to_tensor", type=strtobool, default=False, help="")

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
main_arg_parser.add_argument("--model_filters", type=str, default="[16, 32, 64]", help="")
main_arg_parser.add_argument("--model_classes", type=int, default=2, help="")
main_arg_parser.add_argument("--model_lat_dim", type=int, default=16, help="")
main_arg_parser.add_argument("--model_use_bias", type=strtobool, default=True, help="")
main_arg_parser.add_argument("--model_use_norm", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--model_use_res_net", type=strtobool, default=False, help="")
main_arg_parser.add_argument("--model_dropout", type=float, default=0.00, help="")

# Project
main_arg_parser.add_argument("--project_name", type=str, default='traj-gen', help="")
main_arg_parser.add_argument("--project_owner", type=str, default='si11ium', help="")
main_arg_parser.add_argument("--project_neptune_key", type=str, default=os.getenv('NEPTUNE_KEY'), help="")

# Parse it
args: Namespace = main_arg_parser.parse_args()


def run_lightning_loop(config_obj):

    # Logging
    # ================================================================================
    # Logger
    with Logger(config_obj) as logger:
        # Callbacks
        # =============================================================================
        # Checkpoint Saving
        checkpoint_callback = ModelCheckpoint(
            filepath=str(logger.log_dir / 'ckpt_weights'),
            verbose=True, save_top_k=0,
        )

        # =============================================================================
        # Early Stopping
        # TODO: For This to work, one must set a validation step and End Eval and Score
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=0,
        )

        # Model
        # =============================================================================
        # Init
        model: LightningBaseModule = config_obj.model_class(config_obj.model_paramters)
        model.init_weights(torch.nn.init.xavier_normal_)
        if model.name == 'CNNRouteGeneratorDiscriminated':
            # ToDo: Make this dependent on the used seed
            path = logger.outpath / 'classifier_cnn' / 'version_0'
            disc_model = SavedLightningModels.load_checkpoint(path).restore()
            model.set_discriminator(disc_model)

        # Trainer
        # =============================================================================
        trainer = Trainer(max_epochs=config_obj.train.epochs,
                          show_progress_bar=True,
                          weights_save_path=logger.log_dir,
                          gpus=[0] if torch.cuda.is_available() else None,
                          check_val_every_n_epoch=10,
                          # num_sanity_val_steps=config_obj.train.num_sanity_val_steps,
                          # row_log_interval=(model.n_train_batches * 0.1),  # TODO: Better Value / Setting
                          # log_save_interval=(model.n_train_batches * 0.2),  # TODO: Better Value / Setting
                          checkpoint_callback=checkpoint_callback,
                          logger=logger,
                          fast_dev_run=config_obj.main.debug,
                          early_stop_callback=None
                          )

        # Train It
        trainer.fit(model)

        # Save the last state & all parameters
        trainer.save_checkpoint(logger.log_dir / 'weights.ckpt')
        model.save_to_disk(logger.log_dir)

        # Evaluate It
        if config_obj.main.eval:
            trainer.test()

    return model


if __name__ == "__main__":

    config = Config.read_namespace(args)
    trained_model = run_lightning_loop(config)
