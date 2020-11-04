import torch.utils.data


def create_dataloader(dataset, dataset_opt, use_pin=True):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size_per_month']
        shuffle = dataset_opt['use_shuffle']
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=use_pin)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LR': # Only LR images are provided
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR': # LR and target images are provided
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHR_four_levels': # LR, target with intermediate resolution images are provided
        from data.LRHR_four_levels_dataset import LRHRFourLevelsDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                     dataset_opt['name']))
    return dataset
