from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle
import pandas as pd
import torch

# This will be our class for data loading
class DataLoader(Dataset):
    """Data loader that iterate through a root directory which contains csvs that represent trajectories from day 1 to day 180"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to output arrays from pandas dataframe.
        """
        self.root_dir = root_dir
        self.dirs, self.order = [], []
        for file in os.listdir(self.root_dir):
            if file.endswith(".csv"):
                self.dirs.append(file)
                file = file.replace('output_tick_',"")
                file = file.replace('prog_true',"")
                file = file.replace('progress',"")
                file = file.replace('prog',"")
                f = file.replace('.csv',"")

                self.order.append(int(f))
        self.transform = transform
        self.dirs = ([x for _,x in sorted(zip(self.order,self.dirs))])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.dirs[idx])
        sample = pd.read_csv(file_name, header = None)
        if self.transform:
            sample = self.transform(sample)
        return sample


class Pickler(object):
    """An object to pickle all csv files"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to output arrays from pandas dataframe.
        """
        self.root_dir = root_dir
        for file in os.listdir(self.root_dir):
            if file.endswith(".csv"):
                self.dirs.append(file)
        self.transform = transform

    def __call__(self):
        with tqdm(self.dirs, desc='Pickling') as p:
            i = 0
            for dir in self.dirs:
                file_dir = os.path.join(self.root_dir, dir)
                sample = pd.read_csv(file_dir)
                if self.transform:
                    sample = self.transform(sample)
                file_name = dir.split(".")[0]
                with open(file_name + ".pickle", 'wb') as f:
                    f.dump(sample)
                    f.close()
                p.set_postfix({'Pickled': i+1})


class PickleLoader(Dataset):
    """Data loader that iterate through a root directory which contains pickles that represent trajectories from day 1 to day 180"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to output arrays from pandas dataframe.
        """
        self.root_dir = root_dir
        self.dirs, self.order = [], []
        for file in os.listdir(self.root_dir):
            self.dirs.append(file)
            self.order.append(int(file))

        self.transform = transform
        self.dirs = ([x for _,x in sorted(zip(self.order,self.dirs))])

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(self.root_dir, self.dirs[idx])
        with open(file_name, 'rb') as f:
            sample = pickle.load(f)
        if self.transform:
            sample = self.transform(sample)
        return sample


class Transform(object):
    """
    A transformer that transform a pandas dataframe into a matrix with columns of interests
    """
    def __init__(self, *fields):
        """
        Args:
            fields: list of ints, indicate the index of columns of interests
        """
        self.fields = np.array([*fields], dtype=np.int32)

    def __call__(self, df):
        df = df.values()
        assert all(self.fields < df.shape[1]) and all(self.fields >= 0)
        return df[:, self.fields]
