from base import BaseDataSet
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        self.sampler = None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': False,
            'drop_last': True
        }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)


class CDDataset(BaseDataLoader):
    def __init__(self, kwargs):
        self.batch_size = kwargs.pop('batch_size')
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        
        self.dataset = BaseDataSet(**kwargs)

        super(CDDataset, self).__init__(self.dataset, self.batch_size, shuffle, num_workers)
