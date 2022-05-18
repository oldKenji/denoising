import os
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, data_folder='data', mode='train'):
        super().__init__()

        self.data_folder = data_folder
        self.mode = mode
        self.data = []

        alldata = os.walk(os.path.join(data_folder, mode, 'clean'))
        alldata = [*alldata]

        for dt in alldata[1:]:
            for d in dt[2]:
                self.data.append(d)

    def __getitem__(self, index):

        file_name = self.data[index]
        speaker = file_name.split('_')[0]

        clean_path = os.path.join(self.data_folder, self.mode, 'clean', speaker, file_name)
        noisy_path = os.path.join(self.data_folder, self.mode, 'noisy', speaker, file_name)

        clean = torch.from_numpy(np.load(clean_path))
        noisy = torch.from_numpy(np.load(noisy_path))

        return clean, noisy

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):

        good_clean = []
        good_noisy = []

        for clean, noisy in batch:
            good_clean.append(clean)
            good_noisy.append(noisy)

        good_clean = torch.cat(good_clean, dim=0)
        good_noisy = torch.cat(good_noisy, dim=0)

        add_clean = torch.zeros((int(good_clean.shape[0] / 50) + 1) * 50, 80)
        add_noisy = torch.zeros((int(good_noisy.shape[0] / 50) + 1) * 50, 80)

        for i in range(good_clean.shape[0]):
            add_clean[i] = good_clean[i]
            add_noisy[i] = good_noisy[i]

        return add_clean.reshape(-1, 50, 80).permute(0, 2, 1), add_noisy.reshape(-1, 50, 80).permute(0, 2, 1)
