import warnings

from utils.config import Config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Imports
# =============================================================================

from _templates.new_project.main import run_lightning_loop, args


if __name__ == '__main__':

    # Model Settings
    config = Config().read_namespace(args)
    # bias, activation, model, norm, max_epochs, filters
    cnn_classifier = dict(train_epochs=10, model_use_bias=True, model_use_norm=True, model_activation='leaky_relu',
                          model_type='classifier_cnn', model_filters=[16, 32, 64], data_batchsize=512)
    # bias, activation, model, norm, max_epochs, sr, feature_mixed_dim, filters

    for arg_dict in [cnn_classifier]:
        for seed in range(5):
            arg_dict.update(main_seed=seed)

            config = config.update(arg_dict)

            run_lightning_loop(config)
