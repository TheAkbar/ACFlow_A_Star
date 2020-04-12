
def get_model(hps):
    if hps.model == 'acflow':
        from .acflow import Model
        model = Model(hps)
    elif hps.model == 'acflow_classifier':
        from .acflow_classifier import Model
        model = Model(hps)
    elif hps.model == 'acflow_regressor':
        from .acflow_regressor import Model
        model = Model(hps)
    elif hps.model == 'acflow_sparse':
        from .acflow_sparse import Model
        model = Model(hps)
    else:
        raise Exception()

    return model
    