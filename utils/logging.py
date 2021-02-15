import inspect
from argparse import ArgumentParser
from pathlib import Path

import os
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.api_exceptions import ProjectNotFound

from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.utilities import argparse_utils

from ml_lib.utils.tools import add_argparse_args


class Logger(LightningLoggerBase):

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    media_dir = 'media'

    @classmethod
    def add_argparse_args(cls, parent_parser):
        return add_argparse_args(cls, parent_parser)

    @property
    def experiment(self):
        if self.debug:
            return self.csvlogger.experiment
        else:
            return self.neptunelogger.experiment

    @property
    def log_dir(self):
        return Path(self.csvlogger.experiment.log_dir)

    @property
    def project_name(self):
        return f"{self.owner}/{self.name.replace('_', '-')}"

    @property
    def version(self):
        return self.seed

    @property
    def save_dir(self):
        return self.log_dir

    @property
    def outpath(self):
        return Path(self.root_out) / self.model_name

    def __init__(self, owner, neptune_key, model_name, project_name='', outpath='output', seed=69, debug=False):
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

        self.debug = debug
        self._name = project_name or Path(os.getcwd()).name if not self.debug else 'test'
        self.owner = owner if not self.debug else 'testuser'
        self.neptune_key = neptune_key if not self.debug else 'XXX'
        self.root_out = outpath if not self.debug else 'debug_out'
        self.seed = seed
        self.model_name = model_name

        self._csvlogger_kwargs = dict(save_dir=self.outpath, version=self.version, name=self.name)
        self._neptune_kwargs = dict(offline_mode=self.debug,
                                    api_key=self.neptune_key,
                                    experiment_name=self.name,
                                    project_name=self.project_name)
        try:
            self.neptunelogger = NeptuneLogger(**self._neptune_kwargs)
        except ProjectNotFound as e:
            print(f'The project "{self.project_name}"')
            print(e)

        self.csvlogger = CSVLogger(**self._csvlogger_kwargs)

    def log_hyperparams(self, params):
        self.neptunelogger.log_hyperparams(params)
        self.csvlogger.log_hyperparams(params)
        pass

    def log_metrics(self, metrics, step=None):
        self.neptunelogger.log_metrics(metrics, step=step)
        self.csvlogger.log_metrics(metrics, step=step)
        pass

    def close(self):
        self.csvlogger.close()
        self.neptunelogger.close()

    def log_text(self, name, text, **_):
        # TODO Implement Offline variant.
        self.neptunelogger.log_text(name, text)

    def log_metric(self, metric_name, metric_value, **kwargs):
        self.csvlogger.log_metrics(dict(metric_name=metric_value))
        self.neptunelogger.log_metric(metric_name, metric_value, **kwargs)

    def log_image(self, name, image, ext='png', **kwargs):
        step = kwargs.get('step', None)
        image_name = f'{step}_{name}' if step is not None else name
        image_path = self.log_dir / self.media_dir / f'{image_name}.{ext[1:] if ext.startswith(".") else ext}'
        (self.log_dir / self.media_dir).mkdir(parents=True, exist_ok=True)
        image.savefig(image_path, bbox_inches='tight', pad_inches=0)
        self.neptunelogger.log_image(name, str(image_path), **kwargs)

    def save(self):
        self.csvlogger.save()
        self.neptunelogger.save()

    def finalize(self, status):
        self.csvlogger.finalize(status)
        self.neptunelogger.finalize(status)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize('success')
        pass
