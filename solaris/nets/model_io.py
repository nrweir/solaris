import os
from ..data import zoo_dir
from ..utils.config import _parse_zoo
from tensorflow import keras
import torch
import requests

# below dictionary lists models compatible with solaris. alternatively, your
# own model can be used by using the path to the model as the value for
# model_name in the config file.

model_dict, url_dict = _parse_zoo()


def get_model(model_name, framework, model_path=None, model_url=None):
    """Load a model from a file based on its name."""
    # if the path to the model file isn't provided, get it from model_dict
    if model_path is None:
        model_path = model_dict.get(model_name, None)
    # try to load the model. If it fails to load (hasn't been downloaded yet),
    # download it and save it at model_path, then load it.
    try:
        model = _load_model(framework, model_path)
    except (OSError, FileNotFoundError):
        model_path = download_model(framework, model_name, model_path,
                                    model_url)
        model = _load_model(framework, model_path)

    return model


def _load_model(framework, path):
    """Backend for loading the model."""

    if framework.lower() == 'keras':
        try:
            model = keras.models.load_model(path)
        except OSError:
            raise FileNotFoundError("{} doesn't exist.".format(path))

    elif framework.lower() in ['torch', 'pytorch']:
        # pytorch already throws the right error on failed load, so no need
        # to fix exception
        model = torch.load(path)

    return model


def download_model(framework, model_name, model_path=None,
                   model_url=None, verbose=False):
    """Download model files for training and/or inference.

    Arguments
    ---------
    framework : str
        The deep learning framework the model is associated with. Options are
        ``['keras', 'torch']`` .
    model_name : str
        The string-formatted name of the model, provided in the config file.
    model_path : str, optional
        The path to save the model to. If not provided and the model isn't in
        the core set associated with ``solaris`` , the model will be saved to
        the model zoo at
        ``solaris/data/zoo/[model_name].[framework-specific extension]`` .
    model_url : str, optional
        The URL to the model file to be downloaded. This must be required if
        the model is not in the core set associated with ``solaris`` .
    verbose : bool, optional
        Verbose text output. Defaults to off ( ``False`` ).

    Returns
    -------
    model_path : The path the model is saved to after download.
    """

    # handle bad arguments and fill in missing values first
    if model_url is None and model_name not in list(url_dict.keys()):
        raise ValueError('If using a non-canonical model, the download URL '
                         'must be provided.')
    if model_url is None:
        model_url = url_dict[model_name]
    if model_path is None:
        model_path = model_dict.get(model_name, None)
        if model_path is None:
            if framework == 'keras':
                ext = '.h5'
            elif framework in ['torch', 'pytorch']:
                ext = '.pt'
            model_path = os.path.join(zoo_dir, model_name + ext)

    if verbose:
        print('Beginning model download...')
    r = requests.get(model_url)
    r.raise_for_status()  # raise error if the url is invalid
    with open(model_path, 'wb') as f:
        f.write(r.content)
        f.close()

    return model_path


def reset_weights(model, framework):
    """Re-initialize model weights for training.

    Arguments
    ---------
    model : :class:`tensorflow.keras.Model` or :class:`torch.nn.Module`
        A pre-trained, compiled model with weights saved.
    framework : str
        The deep learning framework used. Currently valid options are
        ``['torch', 'keras']`` .

    Returns
    -------
    reinit_model : model object
        The model with weights re-initialized. Note this model object will also
        lack an optimizer, loss function, etc., which will need to be added.
    """

    if framework == 'keras':
        model_json = model.to_json()
        reinit_model = keras.models.model_from_json(model_json)
    elif framework == 'torch':
        reinit_model = model.apply(_reset_torch_weights)

    return reinit_model


def _reset_torch_weights(torch_layer):
    if isinstance(torch_layer, torch.nn.Conv2d) or \
            isinstance(torch_layer, torch.nn.Linear):
        torch_layer.reset_parameters()
