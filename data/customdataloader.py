from torch.utils.data import DataLoader


class DataLoaders(DataLoader):
    def __init__(self, data_loaders, **kwargs):
        super().__init__(**kwargs)
        self.data_loaders = data_loaders
        self.len = [len(data_loader) for data_loader in self.data_loaders.values()]

    def __getitem__(self, index):
        idx = index
        cnt = 0
        while idx >= self.len[cnt]:
            idx -= self.len[cnt]
            cnt += 1
        return self.data_loaders[cnt][idx]

    def __len__(self):
        return sum(self.len)
