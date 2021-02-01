from abc import ABC
from pathlib import Path

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.api_exceptions import ProjectNotFound
# noinspection PyUnresolvedReferences
from pytorch_lightning.loggers.csv_logs import CSVLogger

from .config import Config


class Logger(LightningLoggerBase, ABC):

    media_dir = 'media'

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
        if self.debug:
            self.config.add_section('project')
            self.config.set('project', 'owner', 'testuser')
            self.config.set('project', 'name', 'test')
            self.config.set('project', 'neptune_key', 'XXX')
        self._csvlogger_kwargs = dict(save_dir=self.outpath, version=self.version, name=self.name)
        self._neptune_kwargs = dict(offline_mode=self.debug,
                                    api_key=self.config.project.neptune_key,
                                    experiment_name=self.name,
                                    project_name=self.project_name,
                                    params=self.config.model_paramters)
        try:
            self.neptunelogger = NeptuneLogger(**self._neptune_kwargs)
        except ProjectNotFound as e:
            print(f'The project "{self.project_name}"')
            print(e)

        self.csvlogger = CSVLogger(**self._csvlogger_kwargs)
        self.log_config_as_ini()

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

    def log_config_as_ini(self):
        self.config.write(self.log_dir / 'config.ini')

    def log_text(self, name, text, step_nb=0, **_):
        # TODO Implement Offline variant.
        self.neptunelogger.log_text(name, text, step_nb)

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
