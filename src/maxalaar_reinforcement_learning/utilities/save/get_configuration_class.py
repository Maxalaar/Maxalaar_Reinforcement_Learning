import importlib.util
from pathlib import Path


def get_configuration_class(configuration_file_path: str):
    configuration_file_path = Path(configuration_file_path).resolve()
    class_name = configuration_file_path.stem

    print('Configuration load : ' + str(configuration_file_path))

    spec = importlib.util.spec_from_file_location(class_name, configuration_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    configuration_class = getattr(module, class_name)

    return configuration_class
