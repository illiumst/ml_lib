##########################
# constants
import argparse
import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np

import pandas as pd

import os


# ToDo: Check this
import shutil
from imageio import imwrite
from natsort import natsorted
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import _logger as log
from test_tube.log import DDPExperiment

_ROOT = Path(os.path.abspath(__file__))


# -----------------------------
# Experiment object
# -----------------------------
class Experiment(object):

    def __init__(self, save_dir=None, name='default', debug=False, version=None, autosave=False, description=None):
        """
        A new Experiment object defaults to 'default' unless a specific name is provided
        If a known name is already provided, then the file version is changed
        :param name:
        :param debug:
        """

        # change where the save dir is if requested

        if save_dir is not None:
            global _ROOT
            _ROOT = save_dir

        self.save_dir = save_dir
        self.no_save_dir = save_dir is None
        self.metrics = []
        self.tags = {}
        self.name = name
        self.debug = debug
        self.version = version
        self.autosave = autosave
        self.description = description
        self.exp_hash = '{}_v{}'.format(self.name, version)
        self.created_at = str(datetime.utcnow())
        self.process = os.getpid()

        # when debugging don't do anything else
        if debug:
            return

        # update version hash if we need to increase version on our own
        # we will increase the previous version, so do it now so the hash
        # is accurate
        if version is None:
            old_version = self.__get_last_experiment_version()
            self.exp_hash = '{}_v{}'.format(self.name, old_version + 1)
            self.version = old_version + 1

        # create a new log file
        self.__init_cache_file_if_needed()

        # when we have a version, load it
        if self.version is not None:

            # when no version and no file, create it
            if not os.path.exists(self.__get_log_name()):
                self.__create_exp_file(self.version)
            else:
                # otherwise load it
                self.__load()

        else:
            # if no version given, increase the version to a new exp
            # create the file if not exists
            old_version = self.__get_last_experiment_version()
            self.version = old_version
            self.__create_exp_file(self.version + 1)

    def get_meta_copy(self):
        """
        Gets a meta-version only copy of this module
        :return:
        """
        return DDPExperiment(self)

    def on_exit(self):
        pass

    def __clean_dir(self):
        files = os.listdir(self.save_dir)

        for f in files:
            if str(self.process) in f:
                os.remove(os.path.join(self.save_dir, f))

    def argparse(self, argparser):
        parsed = vars(argparser)
        to_add = {}

        # don't store methods
        for k, v in parsed.items():
            if not callable(v):
                to_add[k] = v

        self.tag(to_add)

    def add_meta_from_hyperopt(self, hypo):
        """
        Transfers meta data about all the params from the
        hyperoptimizer to the log
        :param hypo:
        :return:
        """
        meta = hypo.get_current_trial_meta()
        for tag in meta:
            self.tag(tag)

    # --------------------------------
    # FILE IO UTILS
    # --------------------------------
    def __init_cache_file_if_needed(self):
        """
        Inits a file that we log historical experiments
        :return:
        """
        try:
            exp_cache_file = self.get_data_path(self.name, self.version)
            if not os.path.isdir(exp_cache_file):
                os.makedirs(exp_cache_file, exist_ok=True)
        except FileExistsError:
            # file already exists (likely written by another exp. In this case disable the experiment
            self.debug = True

    def __create_exp_file(self, version):
        """
        Recreates the old file with this exp and version
        :param version:
        :return:
        """

        try:
            exp_cache_file = self.get_data_path(self.name, self.version)
            # if no exp, then make it
            path = exp_cache_file / 'meta.experiment'
            path.touch(exist_ok=True)

            self.version = version

            # make the directory for the experiment media assets name
            self.get_media_path(self.name, self.version).mkdir(parents=True, exist_ok=True)

        except FileExistsError:
            # file already exists (likely written by another exp. In this case disable the experiment
            self.debug = True

    def __get_last_experiment_version(self):

        exp_cache_file = self.get_data_path(self.name, self.version).parent
        last_version = -1

        version = natsorted([x.name for x in exp_cache_file.iterdir() if 'version_' in x.name])[-1]
        last_version = max(last_version, int(version.split('_')[1]))

        return last_version

    def __get_log_name(self):
        return self.get_data_path(self.name, self.version) / 'meta.experiment'

    def tag(self, tag_dict):
        """
        Adds a tag to the experiment.
        Tags are metadata for the exp.

        >> e.tag({"model": "Convnet A"})

        :param tag_dict:
        :type tag_dict: dict

        :return:
        """
        if self.debug:
            return

        # parse tags
        for k, v in tag_dict.items():
            self.tags[k] = v

        # save if needed
        if self.autosave:
            self.save()

    def log(self, metrics_dict):
        """
        Adds a json dict of metrics.

        >> e.log({"loss": 23, "coeff_a": 0.2})

        :param metrics_dict:

        :return:
        """
        if self.debug:
            return

        new_metrics_dict = metrics_dict.copy()
        for k, v in metrics_dict.items():
            tmp_metrics_dict = new_metrics_dict.pop(k)
            new_metrics_dict.update(tmp_metrics_dict)

        metrics_dict = new_metrics_dict

        # timestamp
        if 'created_at' not in metrics_dict:
            metrics_dict['created_at'] = str(datetime.utcnow())

        self.__convert_numpy_types(metrics_dict)

        self.metrics.append(metrics_dict)

        if self.autosave:
            self.save()

    @staticmethod
    def __convert_numpy_types(metrics_dict):
        for k, v in metrics_dict.items():
            if v.__class__.__name__ == 'float32':
                metrics_dict[k] = float(v)

            if v.__class__.__name__ == 'float64':
                metrics_dict[k] = float(v)

    def save(self):
        """
        Saves current experiment progress
        :return:
        """
        if self.debug:
            return

        # save images and replace the image array with the
        # file name
        self.__save_images(self.metrics)
        metrics_file_path = self.get_data_path(self.name, self.version) / 'metrics.csv'
        meta_tags_path = self.get_data_path(self.name, self.version) / 'meta_tags.csv'

        obj = {
            'name': self.name,
            'version': self.version,
            'tags_path': meta_tags_path,
            'metrics_path': metrics_file_path,
            'autosave': self.autosave,
            'description': self.description,
            'created_at': self.created_at,
            'exp_hash': self.exp_hash
        }

        # save the experiment meta file
        with atomic_write(self.__get_log_name()) as tmp_path:
            with open(tmp_path, 'w') as file:
                json.dump(obj, file, ensure_ascii=False)

        # save the metatags file
        df = pd.DataFrame({'key': list(self.tags.keys()), 'value': list(self.tags.values())})
        with atomic_write(meta_tags_path) as tmp_path:
            df.to_csv(tmp_path, index=False)

        # save the metrics data
        df = pd.DataFrame(self.metrics)
        with atomic_write(metrics_file_path) as tmp_path:
            df.to_csv(tmp_path, index=False)

    def __save_images(self, metrics):
        """
        Save tags that have a png_ prefix (as images)
        and replace the meta tag with the file name
        :param metrics:
        :return:
        """
        # iterate all metrics and find keys with a specific prefix
        for i, metric in enumerate(metrics):
            for k, v in metric.items():
                # if the prefix is a png, save the image and replace the value with the path
                img_extension = None
                img_extension = 'png' if 'png_' in k else img_extension
                img_extension = 'jpg' if 'jpg' in k else img_extension
                img_extension = 'jpeg' if 'jpeg' in k else img_extension

                if img_extension is not None:
                    # determine the file name
                    img_name = '_'.join(k.split('_')[1:])
                    save_path = self.get_media_path(self.name, self.version)
                    save_path = '{}/{}_{}.{}'.format(save_path, img_name, i, img_extension)

                    # save image to disk
                    if type(metric[k]) is not str:
                        imwrite(save_path, metric[k])

                    # replace the image in the metric with the file path
                    metric[k] = save_path

    def __load(self):
        # load .experiment file
        with open(self.__get_log_name(), 'r') as file:
            data = json.load(file)
            self.name = data['name']
            self.version = data['version']
            self.autosave = data['autosave']
            self.created_at = data['created_at']
            self.description = data['description']
            self.exp_hash = data['exp_hash']

        # load .tags file
        meta_tags_path = self.get_data_path(self.name, self.version) / 'meta_tags.csv'
        df = pd.read_csv(meta_tags_path)
        self.tags_list = df.to_dict(orient='records')
        self.tags = {}
        for d in self.tags_list:
            k, v = d['key'], d['value']
            self.tags[k] = v

        # load metrics
        metrics_file_path = self.get_data_path(self.name, self.version) / 'metrics.csv'
        try:
            df = pd.read_csv(metrics_file_path)
            self.metrics = df.to_dict(orient='records')

            # remove nans and infs
            for metric in self.metrics:
                to_delete = []
                for k, v in metric.items():
                    if np.isnan(v) or np.isinf(v):
                        to_delete.append(k)
                for k in to_delete:
                    del metric[k]

        except Exception:
            # metrics was empty...
            self.metrics = []

    def get_data_path(self, exp_name, exp_version):
        """
        Returns the path to the local package cache
        :param exp_name:
        :param exp_version:
        :return:
        Path
        """
        if self.no_save_dir:
            return _ROOT / 'local_experiment_data' / exp_name, f'version_{exp_version}'
        else:
            return _ROOT / exp_name / f'version_{exp_version}'

    def get_media_path(self, exp_name, exp_version):
        """
        Returns the path to the local package cache
        :param exp_version:
        :param exp_name:
        :return:
        """

        return self.get_data_path(exp_name, exp_version) / 'media'

    # ----------------------------
    # OVERWRITES
    # ----------------------------
    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

    def __hash__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)


@contextlib.contextmanager
def atomic_write(dst_path):
    """A context manager to simplify atomic writing.

    Usage:
    >>> with atomic_write(dst_path) as tmp_path:
    >>>     # write to tmp_path
    >>> # Here tmp_path renamed to dst_path, if no exception happened.
    """
    tmp_path = dst_path / '.tmp'
    try:
        yield tmp_path
    except:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        # If everything is fine, move tmp file to the destination.
        shutil.move(tmp_path, str(dst_path))


##########################
class LocalLogger(LightningLoggerBase):

    @property
    def name(self) -> str:
        return self._name

    @property
    def experiment(self) -> Experiment:
        r"""

        Actual TestTube object. To use TestTube features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_test_tube_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description
        )
        return self._experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_hyperparams(self, params: argparse.Namespace):
        pass

    @property
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.save_dir / self.name

        if not root_dir.is_dir():
            log.warning(f'Missing logger folder: {root_dir}')
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def __init__(self, save_dir: str, name: str = "default", description: Optional[str] = None,
                 debug: bool = False, version: Optional[int] = None, **kwargs):
        super(LocalLogger, self).__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self._name = name
        self.description = description
        self.debug = debug
        self._version = version
        self._experiment = None

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state: Dict[Any, Any]):
        self._experiment = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)
