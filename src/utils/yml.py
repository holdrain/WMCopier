import yaml
from yaml import Loader, Dumper
from collections import OrderedDict
from easydict import EasyDict


def load_config(cfgpath):
    Loader, Dumper = OrderedYaml()
    with open(cfgpath, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    return EasyDict(dict_to_nonedict(opt))

def update_config(config,args):
    merged_params = {**dict(config),**vars(args)}
    return EasyDict(merged_params)


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


class NoneDict(dict):
    def __missing__(self, key):
        return None
# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt