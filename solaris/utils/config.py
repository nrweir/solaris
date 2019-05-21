import os
import yaml
from ..data import zoo_dir


def parse(path):
    """Parse a config file for running a model.

    Arguments
    ---------
    path : str
        Path to the YAML-formatted config file to parse.

    Returns
    -------
    config : dict
        A `dict` containing the information from the config file at `path`.

    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    if not config['train'] and not config['infer']:
        raise ValueError('"train", "infer", or both must be true.')
    if config['train'] and config['data']['train_im_src'] is None:
        raise ValueError('"train_im_src" must be provided if training.')
    if config['train'] and config['data']['train_label_src'] is None:
        raise ValueError('"train_label_src" must be provided if training.')
    if config['infer'] and config['data']['infer_im_src'] is None:
        raise ValueError('"infer_im_src" must be provided if "infer".')
    # TODO: IMPLEMENT UPDATING VALUES BASED ON EMPTY ELEMENTS HERE!

    return config


def _parse_zoo():
    """Get the filenames and URLs for all of the models in the zoo."""
    with open(os.path.join(zoo_dir, 'model_reference.yml'), 'r') as f:
        zoo_metadict = yaml.safe_load(f)
        f.close()

    model_paths = zoo_metadict['model_paths']
    model_urls = zoo_metadict['model_urls']

    for model_name, model_path in model_paths.items():
        model_paths[model_name] = os.path.join(zoo_dir, model_path)

    return model_paths, model_urls
