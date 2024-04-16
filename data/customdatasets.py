from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data_sets, **kwargs):
        super().__init__(**kwargs)
        self.data_sets = data_sets
        self.len = [len(data_set) for data_set in self.data_sets]

    def __getitem__(self, index):
        idx = index
        cnt = 0
        while idx >= self.len[cnt]:
            idx -= self.len[cnt]
            cnt += 1
        return self.data_sets[cnt][idx]

    def __len__(self):
        return sum(self.len)
