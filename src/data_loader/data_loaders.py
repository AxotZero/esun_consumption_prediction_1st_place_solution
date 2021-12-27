from pdb import set_trace as bp

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np

from base import BaseDataLoader
from constant import target_indices


# def padding_mask_collate(batch):
#     """
#     return 
#     """


def batch_index_collate(data):
    data = list(zip(*data))
    y = torch.stack(data[1], 0)
    
    batch_indices = []
    for i, d in enumerate(data[0]):
        batch_indices += [i] * int(d.size()[0])
        
    return (
        (torch.Tensor(batch_indices), torch.cat(data[0]).float()), 
        y.float()
    )


class MultiIndexDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, 
                     data_path, 
                     training=True):
            self.load_xs(data_path)
            self.training = training
            print(f'=== dataloader mode is {"training" if self.training else "testing"} ===')
            self.chids = list(self.xs.keys())

        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            return len(self.chids)

        def get_customer(self, chid):
            x = self.xs[chid]
            # 0: dt, 1: shop_tag, 2: txn_cnt, 3: txn_amt
            x[:,0] = x[:,0] - 1

            # create y
            y = torch.zeros(24*49, dtype=torch.float)
            indices = (x[:, 0] * 49 + x[:, 1]).astype(int)
            y[indices] = torch.tensor(x[:, -1], dtype=torch.float)
            y = y.view(24, 49)

            return (
                torch.tensor(x),
                y[1:]
            )

        def __getitem__(self, i):
            chid = self.chids[i]
            x, y = self.get_customer(chid)
            if self.training:
                return x, y
            else:
                return x, torch.tensor(chid)

    def __init__(self, 
                 data_path,
                 batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.dataset = self.__class__.InnerDataset(
            data_path,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, fold_idx, validation_split, num_workers, collate_fn=batch_index_collate)

    
class Seq2SeqWithDtDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, 
                     data_path, 
                     num_classes=49, 
                     training=True):
            self.load_xs(data_path)
            self.num_classes = num_classes
            self.n_month = 24
            self.nrows_per_month = 26
            self.training = training
            print(f'=== dataloader mode is {"training" if self.training else "testing"} ===')
            self.chids = list(self.xs.keys())

        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            return len(self.chids)

        def get_customer(self, chid):
            x = self.xs[chid]
            # 0: dt, 1: shop_tag, 2: txn_cnt, 3: txn_amt
            x[:,0] = x[:,0] - 1

            # generate data indices and mask
            values, counts = np.unique(x[:,0], return_counts=True)
            counts_indices = np.concatenate([list(range(count)) for count in counts])
            indices = (x[:, 0] * self.nrows_per_month + counts_indices).astype(int)
            rows_per_month_mask = np.ones(self.n_month*self.nrows_per_month )
            rows_per_month_mask[indices] = 0
            month_mask = np.ones(self.n_month)
            month_mask[values.astype(int)] = 0

            # x mapping
            ret = np.zeros((self.n_month*self.nrows_per_month, x.shape[-1]), dtype=float)
            ret[indices] = x
            ret[:, 0] = ret[:, 0] % 12 # modulo dt

            # create y
            y = torch.zeros(self.n_month*49, dtype=torch.float)
            indices = (x[:, 0] * 49 + x[:, 1]).astype(int)
            y[indices] = torch.tensor(x[:, -1], dtype=torch.float)
            y = y.view(self.n_month, 49)
            if self.num_classes == 16:
                y = y[:, target_indices]

            return (
                (
                    torch.tensor(ret.reshape(self.n_month, self.nrows_per_month, -1), dtype=torch.float),
                    torch.tensor(rows_per_month_mask.reshape(self.n_month, self.nrows_per_month), dtype=torch.bool),
                    torch.tensor(month_mask, dtype=torch.bool),
                ),
                y[1:]
            )

        def __getitem__(self, i):
            chid = self.chids[i]
            x, y = self.get_customer(chid)
            if self.training:
                return x, y
            else:
                return x, torch.tensor(chid)

    def __init__(self, 
                 data_path,
                 num_classes=49, 
                 batch_size=128, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.dataset = self.__class__.InnerDataset(
            data_path,
            num_classes=num_classes,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Seq2SeqDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, 
                     data_path, 
                     num_classes=49, 
                     training=True):
            self.load_xs(data_path)
            self.num_classes = num_classes
            self.n_month = 24
            self.nrows_per_month = 26
            self.training = training
            print(f'=== dataloader mode is {"training" if self.training else "testing"} ===')
            self.chids = list(self.xs.keys())

        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            return len(self.chids)

        def get_customer(self, chid):
            x = self.xs[chid]
            # 0: dt, 1: shop_tag, 2: txn_cnt, 3: txn_amt
            x[:,0] = x[:,0] - 1

            # generate data indices and mask
            values, counts = np.unique(x[:,0], return_counts=True)
            counts_indices = np.concatenate([list(range(count)) for count in counts])
            indices = (x[:, 0] * self.nrows_per_month + counts_indices).astype(int)
            rows_per_month_mask = np.ones(self.n_month*self.nrows_per_month )
            rows_per_month_mask[indices] = 0
            month_mask = np.ones(self.n_month)
            month_mask[values.astype(int)] = 0

            # x mapping
            ret = np.zeros((self.n_month*self.nrows_per_month, x.shape[-1]-1), dtype=float)
            ret[indices] = x[:, 1:]
            # create y
            y = torch.zeros(self.n_month*49, dtype=torch.float)
            indices = (x[:, 0] * 49 + x[:, 1]).astype(int)
            y[indices] = torch.tensor(x[:, -1], dtype=torch.float)
            y = y.view(self.n_month, 49)
            if self.num_classes == 16:
                y = y[:, target_indices]

            return (
                (
                    torch.tensor(ret.reshape(self.n_month, self.nrows_per_month, -1), dtype=torch.float),
                    torch.tensor(rows_per_month_mask.reshape(self.n_month, self.nrows_per_month), dtype=torch.bool),
                    torch.tensor(month_mask, dtype=torch.bool),
                ),
                y[1:]
            )

        def __getitem__(self, i):
            chid = self.chids[i]
            x, y = self.get_customer(chid)
            if self.training:
                return x, y
            else:
                return x, torch.tensor(chid)

    def __init__(self, 
                 data_path,
                 num_classes=49, 
                 batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.dataset = self.__class__.InnerDataset(
            data_path,
            num_classes=num_classes,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, fold_idx, validation_split, num_workers)


class BigArchMaskDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, 
                     data_path, 
                     target_pkl_path, 
                     train_month_range=(1, 23), 
                     test_month_range=(1,24),
                     num_classes=49, 
                     training=True):
            self.load_xs(data_path)
            self.num_classes = num_classes
            self.nrows_per_month = 26
            self.training = training
            if training:
                self.start_dt, self.end_dt = train_month_range
                self.target_list = pickle.load(open(target_pkl_path, 'rb'))
                print('=== dataloader mode is training ===')
            else:
                self.start_dt, self.end_dt = test_month_range
                self.chids = list(self.xs.keys())
                print('=== dataloader mode is testing ===')
            self.n_month = self.end_dt - self.start_dt + 1
            print(f"start_dt: {self.start_dt}, end_dt: {self.end_dt}")


        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            if self.training:
                return len(self.target_list)
            else:
                return len(self.chids)

        def get_customer(self, chid, start_dt=1, end_dt=24):
            x = self.xs[chid]
            # 0: dt, 1: shop_tag, 2: txn_cnt, 3: txn_amt
            
            x = x[(x[:, 0] <= end_dt) & (x[:, 0] >= start_dt)]
            x[:,0] = x[:,0] - start_dt

            # generate data indices and mask
            values, counts = np.unique(x[:,0], return_counts=True)
            indices = np.concatenate([list(range(count)) for count in counts] )
            indices = (x[:, 0] * self.nrows_per_month + indices).astype(int)
            rows_per_month_mask = np.ones(self.n_month*self.nrows_per_month )
            rows_per_month_mask[indices] = 0
            month_mask = np.ones(self.n_month)
            month_mask[values.astype(int)] = 0
            # x mapping
            ret = np.zeros((self.n_month*self.nrows_per_month, x.shape[-1]-1), dtype=float)
            ret[indices] = x[:, 1:]
            return (
                torch.tensor(ret.reshape(self.n_month, self.nrows_per_month, -1), dtype=torch.float),
                torch.tensor(rows_per_month_mask.reshape(self.n_month, self.nrows_per_month), dtype=torch.bool),
                torch.tensor(month_mask, dtype=torch.bool),
            )

        def __getitem__(self, i):
            if self.training:
                """
                return x, y
                x = customer[:-1], y = customer[-1].shop_tag / txn_amt, index is month
                """
                # get target
                chid, target_dt, target_class, target_txn_amt = self.target_list[i]
                x = self.get_customer(chid, self.start_dt, target_dt-1)
                
                # process y
                y = torch.zeros(49, dtype=torch.float)
                y[target_class] = torch.tensor(
                    target_txn_amt, dtype=torch.float) / sum(target_txn_amt)
                if self.num_classes == 16:
                    y = y[target_indices]
                return x, y
            else:
                """
                return x, chid
                x = customer[1:]
                """
                chid = self.chids[i]
                x = self.get_customer(chid, self.start_dt, self.end_dt)
                return x, torch.tensor(chid)

    def __init__(self, 
                 data_path, 
                 target_pkl_path, 
                 train_month_range=(1,23), 
                 test_month_range=(1,24),
                 num_classes=49, 
                 batch_size=128, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.target_pkl_path = target_pkl_path
        self.dataset = self.__class__.InnerDataset(
            data_path,
            target_pkl_path,
            train_month_range=train_month_range,
            test_month_range=test_month_range,
            num_classes=num_classes,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BigArchDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, data_path, target_pkl_path, dt_len=23, month_row_num=10, training=True):
            self.load_xs(data_path)
            self.dt_len = dt_len
            self.month_row_num = month_row_num
            self.training = training
            if training:
                self.target_list = pickle.load(open(target_pkl_path, 'rb'))
                print('=== dataloader mode is training ===')
            else:
                # self.dt_len = 24
                self.chids = list(self.xs.keys())
                print('=== dataloader mode is testing ===')


        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            if self.training:
                return len(self.target_list)
            else:
                return len(self.chids)

        def get_customer(self, chid, start_dt=0, end_dt=24):
            x = pd.DataFrame(self.xs[chid])
            # 0: dt, 1: shop_tag, 2: txn_cnt, 3: txn_amt
            x = x[(x[0] <= end_dt) & (x[0] >= start_dt)]
            # get topk
            if self.month_row_num < 26:
                x = x.groupby(0).apply(lambda x: x.nlargest(
                    self.month_row_num, [3])).reset_index(drop=True)
            ret = np.zeros((self.dt_len, self.month_row_num, x.shape[-1]-1), dtype=float)
            for month, df in x.groupby(0):
                ret[int(month-start_dt)][:len(df)] = df.drop(columns=[0]).values
            return torch.tensor(ret, dtype=torch.float)

        def __getitem__(self, i):
            if self.training:
                """
                return x, y
                x = customer[:-1], y = customer[-1].shop_tag / txn_amt, index is month
                """
                # get target
                chid, target_dt, target_class, target_txn_amt = self.target_list[i]
                x = self.get_customer(chid, start_dt=1, end_dt=target_dt-1)

                y = torch.zeros(49, dtype=torch.float)
                # process y
                y[target_class] = torch.tensor(
                    target_txn_amt, dtype=torch.float) / sum(target_txn_amt)
                return x, y
            else:
                """
                return x, chid
                x = customer[1:]
                """
                chid = self.chids[i]
                x = self.get_customer(chid, start_dt=2, end_dt=24)
                return x, torch.tensor(chid)

    def __init__(self, data_path, target_pkl_path, dt_len=23, month_row_num=10, batch_size=128, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.target_pkl_path = target_pkl_path
        self.dt_len = dt_len
        self.dataset = self.__class__.InnerDataset(
            data_path,
            target_pkl_path,
            dt_len=dt_len,
            month_row_num=month_row_num,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CCDataLoader2(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, data_path, target_pkl_path, dt_len=23, padding=440, training=True):
            self.load_xs(data_path)
            self.target_list = pickle.load(open(target_pkl_path, 'rb'))
            self.dt_len = dt_len
            self.padding = padding
            self.training = training
            if not training:
                self.chids = list(self.xs.keys())
                print('=== dataloader mode is testing ===')

        def load_xs(self, data_path):
            """
            xs is a map which key is chid and value is dataframe
            """
            print('start load_xs')
            self.xs = pickle.load(open(data_path, 'rb'))
            print('finish load_xs!')

        def __len__(self):
            if self.training:
                return len(self.target_list)
            else:
                return len(self.chids)

        def __getitem__(self, i):
            if self.training:
                """
                return x, y
                x = customer[:-1], y = customer[-1].shop_tag / txn_amt, index is month
                """
                # get target
                chid, target_dt, target_class, target_txn_amt = self.target_list[i]

                x = np.copy(self.xs[chid])
                x = x[(x[:, 0] < target_dt)]
                x[:, 0] = (target_dt - x[:, 0]) // self.dt_len
                x = F.pad(torch.tensor(x, dtype=torch.float),
                          (0, 0, 0, self.padding-x.shape[0]), value=0)

                y = torch.zeros(49, dtype=torch.float)
                # process y
                y[target_class] = torch.tensor(
                    target_txn_amt, dtype=torch.float) / sum(target_txn_amt)
                return x, y
            else:
                """
                return x, chid
                x = customer[1:]
                """
                chid = self.chids[i]
                x = np.copy(self.xs[chid])
                x = x[x[:, 0] > 1]
                x[:, 0] = (25 - x[:, 0]) // self.dt_len
                x = F.pad(torch.tensor(x, dtype=torch.float),
                          (0, 0, 0, self.padding-x.shape[0]), value=0)
                return x, torch.tensor(chid)

    def __init__(self, data_path, target_pkl_path, dt_len=23, padding=440, batch_size=128, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_path = data_path
        self.target_pkl_path = target_pkl_path
        self.dt_len = dt_len
        self.padding = padding
        self.dataset = self.__class__.InnerDataset(
            data_path,
            target_pkl_path,
            dt_len=dt_len,
            padding=padding,
            training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
