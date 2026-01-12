import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr

class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution , cut=False):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.cut =cut
        self.label_set = set(self.label)
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1

        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])
        # print(self.index_dict.shape)
        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i
        # print(self.index_dict)
    def load_all_data(self):
        # print(self.cache)
        for i in range(self.data_size):
            if i % 10000 ==0:
                print('number-',i)
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        if self.cut:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0

        else:
            a = self.img2xarray(
                path).astype('float32') / 255.0  
            return a          
        # Random Horizontal Flip Augmentation (Sequence-level)
        # Only during training (we can check if 'train' is in the path or just assume based on usage)
        do_flip = False
        if self.cache is False: # Simple heuristic: if not caching, it's likely training with random sampling
             if np.random.random() > 0.5:
                 do_flip = True

        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            if do_flip:
                data = [f[:, :, ::-1] for f in data] # Flip width dimension
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index],

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        
        frame_list = []
        for _img_path in imgs:
            img_p = osp.join(flie_path, _img_path)
            if osp.isfile(img_p):
                # Read as grayscale directly
                img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to the required resolution
                    img = cv2.resize(img, (self.resolution, self.resolution))
                    frame_list.append(img)

        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)