import argparse
from typing import Union, Dict, Optional, Any

from abc import ABC
from pathlib import Path

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.test_tube import TestTubeLogger

from .config import Config


class Logger(LightningLoggerBase, ABC):

    media_dir = 'media'

    @property
    def experiment(self):
        if self.debug:
            return self.testtubelogger.experiment
        else:
            return self.neptunelogger.experiment

    @property
    def log_dir(self):
        return Path(self.testtubelogger.experiment.get_logdir()).parent

    @property
    def name(self):
        return self.config.name

    @property
    def project_name(self):
        return f"{self.config.project.owner}/{self.config.project.name.replace('_', '-')}"

    @property
    def version(self):
        return self.config.get('main', 'seed')

    @property
    def outpath(self):
        return Path(self.config.train.outpath) / self.config.model.type

    @property
    def exp_path(self):
        return Path(self.outpath) / self.name

    def __init__(self, config: Config):
        """
        params (dict|None): Optional. Parameters of the experiment. After experiment creation params are read-only.
           Parameters are displayed in the experiment’s Parameters section and each key-value pair can be
           viewed in experiments view as a column.
        properties (dict|None): Optional default is {}. Properties of the experiment.
           They are editable after experiment is created. Properties are displayed in the experiment’s Details and
           each key-value pair can be viewed in experiments view as a column.
        tags (list|None): Optional default []. Must be list of str. Tags of the experiment.
           They are editable after experiment is created (see: append_tag() and remove_tag()).
           Tags are displayed in the experiment’s Details and can be viewed in experiments view as a column.
        """
        super(Logger, self).__init__()

        self.config = config
        self.debug = self.config.main.debug
        self._testtube_kwargs = dict(save_dir=self.outpath, version=self.version, name=self.name)
        self._neptune_kwargs = dict(offline_mode=self.debug,
                                    api_key=self.config.project.neptune_key,
                                    experiment_name=self.name,
                                    project_name=self.project_name,
                                    params=self.config.model_paramters)
        self.neptunelogger = NeptuneLogger(**self._neptune_kwargs)
        self.testtubelogger = TestTubeLogger(**self._testtube_kwargs)
        self.log_config_as_ini()

    def log_hyperparams(self, params):
        self.neptunelogger.log_hyperparams(params)
        self.testtubelogger.log_hyperparams(params)
        pass

    def log_metrics(self, metrics, step=None):
        self.neptunelogger.log_metrics(metrics, step=step)
        self.testtubelogger.log_metrics(metrics, step=step)
        pass

    def close(self):
        self.testtubelogger.close()
        self.neptunelogger.close()

    def log_config_as_ini(self):
        self.config.write(self.log_dir / 'config.ini')

    def log_text(self, name, text, step_nb=0, **kwargs):
        # TODO Implement Offline variant.
        self.neptunelogger.log_text(name, text, step_nb)

    def log_metric(self, metric_name, metric_value, **kwargs):
        self.testtubelogger.log_metrics(dict(metric_name=metric_value))
        self.neptunelogger.log_metric(metric_name, metric_value, **kwargs)

    def log_image(self, name, image, **kwargs):
        self.neptunelogger.log_image(name, image, **kwargs)
        step = kwargs.get('step', None)
        name = f'{step}_{name}' if step is not None else name
        image.savefig(self.log_dir / self.media_dir / name)

    def save(self):
        self.testtubelogger.save()
        self.neptunelogger.save()

    def finalize(self, status):
        self.testtubelogger.finalize(status)
        self.neptunelogger.finalize(status)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize('success')
        pass
