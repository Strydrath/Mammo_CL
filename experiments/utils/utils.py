from types import SimpleNamespace

def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        result = SimpleNamespace(**vars(args), **vars(additional_args))
        print(vars(result))
        return result
    return args