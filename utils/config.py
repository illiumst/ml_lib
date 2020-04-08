import ast

from argparse import Namespace
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

from ml_lib.models.generators.cnn import CNNRouteGeneratorModel
from ml_lib.models.generators.cnn_discriminated import CNNRouteGeneratorDiscriminated

from ml_lib.models.homotopy_classification.cnn_based import ConvHomDetector
from ml_lib.utils.model_io import ModelParameters
from ml_lib.utils.transforms import AsArray


def is_jsonable(x):
    import json
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


class Config(ConfigParser):

    # TODO: Do this programmatically; This did not work:
    # Initialize Default Sections
    # for section in self.default_sections:
    #     self.__setattr__(section, property(lambda x :x._get_namespace_for_section(section))

    @property
    def model_class(self):
        model_dict = dict(ConvHomDetector=ConvHomDetector,
                          CNNRouteGenerator=CNNRouteGeneratorModel,
                          CNNRouteGeneratorDiscriminated=CNNRouteGeneratorDiscriminated
                          )
        try:
            return model_dict[self.get('model', 'type')]
        except KeyError as e:
            raise KeyError(rf'The model alias you provided ("{self.get("model", "type")}") does not exist! \n'
                           f'Try one of these:\n{list(model_dict.keys())}')

    @property
    def main(self):
        return self._get_namespace_for_section('main')

    @property
    def model(self):
        return self._get_namespace_for_section('model')

    @property
    def train(self):
        return self._get_namespace_for_section('train')

    @property
    def data(self):
        return self._get_namespace_for_section('data')

    @property
    def project(self):
        return self._get_namespace_for_section('project')
    ###################################################

    @property
    def model_paramters(self):
        return ModelParameters(self.model, self.train, self.data)

    @property
    def tags(self, ):
        return [f'{key}: {val}' for key, val in self.serializable.items()]

    @property
    def serializable(self):
        return {f'{section}_{key}': val for section, params in self._sections.items()
                for key, val in params.items() if is_jsonable(val)}

    @property
    def as_dict(self):
        return self._sections

    def _get_namespace_for_section(self, item):
        return Namespace(**{key: self.get(item, key) for key in self[item]})

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        pass

    @staticmethod
    def _sort_combined_section_key_mapping(dict_obj):
        sorted_dict = defaultdict(dict)
        for key in dict_obj:
            section, *attr_name = key.split('_')
            attr_name = '_'.join(attr_name)
            value = str(dict_obj[key])

            sorted_dict[section][attr_name] = value
        # noinspection PyTypeChecker
        return dict(sorted_dict)

    @classmethod
    def read_namespace(cls, namespace: Namespace):

        sorted_dict = cls._sort_combined_section_key_mapping(namespace.__dict__)
        new_config = cls()
        new_config.read_dict(sorted_dict)
        return new_config

    def update(self, mapping):
        sorted_dict = self._sort_combined_section_key_mapping(mapping)
        for section in sorted_dict:
            if self.has_section(section):
                pass
            else:
                self.add_section(section)
            for option, value in sorted_dict[section].items():
                self.set(section, option, value)
        return self

    def get(self, *args, **kwargs):
        item = super(Config, self).get(*args, **kwargs)
        try:
            return ast.literal_eval(item)
        except SyntaxError:
            return item
        except ValueError:
            return item

    def write(self, filepath, **kwargs):
        path = Path(filepath, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as configfile:
            super().write(configfile)
        return True
