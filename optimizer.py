import torch.optim as torch_optim


def initialize_optimizer(config, model):
    opt_type = config.opt.lower()
    decay_rate = config.weight_decay
    model_params = model.parameters()

    optim_args = {'lr': config.lr, 'weight_decay': decay_rate}

    split_opt_type = opt_type.split('_')
    opt_type = split_opt_type[-1]

    if opt_type in ['sgd', 'nesterov']:
        optim_args.pop('eps', None)
        optimizer = torch_optim.SGD(model_params, momentum=config.momentum, nesterov=(opt_type == 'nesterov'), **optim_args)
    elif opt_type == 'momentum':
        optim_args.pop('eps', None)
        optimizer = torch_optim.SGD(model_params, momentum=config.momentum, nesterov=False, **optim_args)
    elif opt_type == 'adam':
        optimizer = torch_optim.Adam(model_params, **optim_args)
    elif opt_type == 'adamw':
        optimizer = torch_optim.AdamW(model_params, **optim_args)
    elif opt_type == 'nadam':
        optimizer = torch_optim.Adadelta(model_params, **optim_args)
        optimizer = torch_optim.RMSprop(model_params, alpha=0.9, momentum=config.momentum, **optim_args)
    else:
        raise ValueError("Invalid optimizer type specified")

    with open("output.log", "a+") as log_file:
        print("optimizer is", optimizer, file=log_file)
    
    return optimizer
