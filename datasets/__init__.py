
def get_dataset(split, hps):
    if hps.dataset == 'maf':
        from .maf import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    elif hps.dataset == 'cube':
        from .cube import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size, hps.mask)
    elif hps.dataset == 'uci_reg':
        from .uci_reg import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size, hps.mask)
    else:
        raise Exception()

    assert dataset.d == hps.dimension

    return dataset