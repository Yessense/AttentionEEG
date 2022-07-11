import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, IterableDataset
import scipy.signal as sn
import time
from sklearn.preprocessing import LabelEncoder


def roll2d(a, b, dx=1, dy=1):
    """
    rolling 2d window for nd array
    last 2 dimensions
    parameters
    ----------
    a : ndarray
        target array where is needed rolling window
    b : tuple
        window array size-like rolling window
    dx : int
        horizontal step, abscissa, number of columns
    dy : int
        vertical step, ordinate, number of rows
    returns
    -------
    out : ndarray
        returned array has shape 4
        first two dimensions have size depends on last two dimensions target array
    """
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b[-2]) // dy + 1,) + \
            ((a.shape[-1] - b[-1]) // dx + 1,) + \
            b  # sausage-like shape with 2d cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def roll(a, b, dx=1):
    """
    Rolling 1d window on array
    Parameters
    ----------
    a : ndarray
    b : ndarray
        rolling 1D window array. Example np.zeros(64)
    dx : step size (horizontal)
    Returns
    -------
    out : ndarray
        target array
    """
    shape = a.shape[:-1] + (int((a.shape[-1] - b.shape[-1]) / dx) + 1,) + b.shape
    strides = a.strides[:-1] + (a.strides[-1] * dx,) + a.strides[-1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def abs_fft(x):
    return np.abs(np.fft.rfft(x))


class DatasetCreator():
    def __init__(self,
                 path_to_dir: str,
                 used_columns: Optional[List] = None,
                 target_column: str = 'state',
                 dt: int = 256,
                 bci_exp_numbers=(0, 1, 2, 3, 4, 5),
                 val_exp_numbers: Optional[List[int]] = None,
                 used_classes=(1, 2, 3)):  # , 3)):
        if used_columns is None:
            used_columns = ['F3', 'Fz', 'F4',
                            'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
                            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                            'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
                            'P3', 'Pz', 'P4'
                            ]
        self.le = LabelEncoder()
        self.le.fit(used_classes)
        self.path_to_dir = path_to_dir
        self.used_columns = used_columns
        self.target_column = target_column
        self.dt = dt
        self.s_rate = 128
        self.b, self.a = sn.butter(2, [5, 36], btype='bandpass', fs=self.s_rate)
        self.b37, self.a37 = sn.butter(2, [37, 50], btype='bandstop', fs=self.s_rate)
        self.b50, self.a50 = sn.butter(2, [48, 52], btype='bandstop', fs=self.s_rate)
        self.b60, self.a60 = sn.butter(2, [58, 62], btype='bandstop', fs=self.s_rate)

        self.session_template = "session_{}"
        self.bci_exp_template = "bci_exp_{}"
        self.bci_exp_data = "data.csv"
        self.bci_exp_numbers = bci_exp_numbers
        self.val_exp_numbers = val_exp_numbers

        self.bci_exp_filename = "data.csv"
        self.used_columns = used_columns
        self.used_classes = used_classes

    def create_dataset(self, session_numbers: List[int],
                       shift: int = 128,
                       validation: bool = False):
        sessions_encoder = LabelEncoder()
        sessions_encoder.fit(session_numbers)

        start_time = time.time()
        print(f"-" * 40)
        print(f"Creating dataset with parameters:")
        print(f"\tsession_numbers: {session_numbers}")
        print(f"\tshift: {shift}")
        print(f"\tdt: {self.dt}")
        print(f"\tvalidation: {validation}")
        if validation:
            if self.val_exp_numbers is not None:
                bci_exp_numbers = self.val_exp_numbers
            else:
                bci_exp_numbers = list(set(self.bci_exp_numbers) - set(self.val_exp_numbers))
        else:
            bci_exp_numbers = self.bci_exp_numbers

        dataset_hash = hash(frozenset(session_numbers)) + hash(shift) + hash(validation) + hash(
            frozenset(bci_exp_numbers))
        dataset_hash = str(dataset_hash)

        path_to_dataset = os.path.join(self.path_to_dir, dataset_hash)
        if os.path.exists(path_to_dataset):
            return torch.load(path_to_dataset)

        channels_data = []
        label_data = []
        person_data = []
        fft_data = []

        for session in session_numbers:
            for bci_exp_number in bci_exp_numbers:
                session_name = self.session_template.format(session)
                bci_exp_name = self.bci_exp_template.format(bci_exp_number)
                bci_exp_path = os.path.join(self.path_to_dir, session_name, bci_exp_name, self.bci_exp_data)

                experiment_data = pd.read_csv(bci_exp_path)
                experiment_data = experiment_data[self.used_columns + [self.target_column]]
                experiment_data = experiment_data.to_numpy()

                x = experiment_data[:, :-1]
                x = sn.lfilter(self.b, self.a, x, axis=0)
                x = sn.lfilter(self.b37, self.a37, x, axis=0)
                x = sn.lfilter(self.b50, self.a50, x, axis=0)
                x = sn.lfilter(self.b60, self.a60, x, axis=0)

                mean = x.mean(axis=0)[np.newaxis, :]
                std = x.std(axis=0)[np.newaxis, :]
                x -= mean
                x /= std
                x = roll2d(x, (self.dt, len(self.used_columns)), 1, shift).squeeze()
                x = x.transpose(0, 2, 1)

                y = experiment_data[:, -1]
                y = y[:y.shape[0] - self.dt + 1: shift]

                class_change = np.convolve(experiment_data[:, -1], [1, -1], 'same') != 0
                class_change = np.roll(class_change.astype(np.int32), -1)
                class_change = np.convolve(class_change, [1, 1], 'same')
                conv = np.sum(roll(class_change, np.ones(self.dt), shift), axis=1)
                mask = conv < 2

                used_classes_mask = np.zeros_like(y, dtype=bool)
                for used_class in self.used_classes:
                    used_classes_mask |= y == used_class

                final_mask = mask & used_classes_mask

                channels_data.append(x[final_mask])
                label_data.append(self.le.transform(y[final_mask]))
                person = np.ones_like(y[final_mask], dtype=int) * session
                person = sessions_encoder.transform(person)
                person_data.append(person)
                fft_data.append(np.apply_along_axis(abs_fft, 2, x[final_mask]))


        print(f"Dataset is created. Time elapsed: {time.time() - start_time:0.1f} s.")
        print()
        dataset = torch.tensor(np.concatenate(channels_data, axis=0)).float(), \
                    torch.tensor(np.concatenate(fft_data, axis=0)).float(), \
                  torch.tensor(np.concatenate(label_data, axis=0)).long(), \
                  torch.tensor(np.concatenate(person_data, axis=0)).long()

        torch.save(dataset, path_to_dataset)

        return dataset


class Physionet(Dataset):
    def __init__(self, data: torch.Tensor,
                 fft: torch.Tensor,
                 label_target: torch.Tensor,
                 person_target: torch.Tensor):
        assert data.shape[0] == label_target.shape[0]
        self.fft = fft
        self.size = data.shape[0]
        self.data = data
        self.label_target = label_target
        self.person_target = person_target

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.fft[idx], self.label_target[idx], self.person_target[idx]


if __name__ == '__main__':
    path_to_directory = "/home/yessense/PycharmProjects/eeg_project/data_physionet"
    creator = DatasetCreator(path_to_dir=path_to_directory)
    dataset = Physionet(*creator.create_dataset([1, 5, 6]))

    raw, fft, label, person = next(iter(dataset))
    print(raw.shape)
    print(fft.shape)
    print(label.shape)
    print(person.shape)

    print("Done")
