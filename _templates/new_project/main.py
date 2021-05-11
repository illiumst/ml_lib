# Imports
# =============================================================================

import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ml_lib.modules.util import LightningBaseModule
from ml_lib.utils.config import Config
from ml_lib.utils.loggers import LightningLogger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def run_lightning_loop(config_obj):

    # Logging
    # ================================================================================
    # Logger
    with LightningLogger(config_obj) as logger:
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
        trainer.save_checkpoint(config_obj.exp_path.log_dir / 'weights.ckpt')
        model.save_to_disk(config_obj.exp_path)

        # Evaluate It
        if config_obj.main.eval:
            trainer.test()

    return model


if __name__ == "__main__":
    from _templates.new_project._parameters import args
    config = Config.read_namespace(args)
    trained_model = run_lightning_loop(config)
