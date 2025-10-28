import os.path
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import scipy
import torch
import neurokit2 as nk


class RGBDatasetBase(Dataset):
    def __init__(self, train_list: List[Tuple[str, str]], test_list: List[Tuple[str, str]],
                 num_frame: int, fs: int, multiple: int, is_train: bool):
        self._num_frame = num_frame
        self._fs = fs
        self._multiple = multiple
        self._is_train = is_train
        self._train_list = train_list
        self._test_list = test_list
        self._id2range = list()
        self._num_clip_list = list()
        self._id2sampleid = dict()
        if self._is_train:
            for i in range(len(train_list)):
                length = np.min(
                    [np.load(train_list[i][0]).shape[0], np.load(train_list[i][1]).shape[0]]) - self._num_frame
                if length<=0:
                    continue
                length_avg = length // self._multiple
                start = 0
                end = length_avg
                while end <= length:
                    if end + length_avg > length:
                        end = length
                    self._id2range.append((i, start, end))
                    start += length_avg
                    end += length_avg
        if not self._is_train:
            for i in range(len(test_list)):
                num_clip = np.min([np.load(test_list[i][0]).shape[0], np.load(test_list[i][1]).shape[0]]) // num_frame
                self._num_clip_list.append(num_clip)
            for i in range(sum(self._num_clip_list)):
                current_clips = 0
                for j in range(len(self._num_clip_list)):
                    next_clips = current_clips + self._num_clip_list[j]
                    if i < next_clips:
                        self._id2sampleid[i] = (j, i - current_clips)
                        break
                    else:
                        current_clips = next_clips

    def __len__(self):
        if self._is_train:
            return len(self._id2range)
        else:
            return sum(self._num_clip_list)

    def __getitem__(self, i: int):
        work_list = self._train_list if self._is_train else self._test_list
        if self._is_train:
            sampleid, start, end = self._id2range[i]
            imgs, bvp = np.load(work_list[sampleid][0],mmap_mode='r'), np.load(work_list[sampleid][1])
            num_frame, height, width, num_channel = imgs.shape

            sample_start = np.random.choice(np.arange(start, end))
            sample_end = sample_start + self._num_frame
        else:
            sampleid, innerid = self._id2sampleid[i]
            imgs, bvp = np.load(work_list[sampleid][0],mmap_mode='r'), np.load(work_list[sampleid][1])
            num_frame, height, width, num_channel = imgs.shape
            assert height == width
            sample_start = innerid * self._num_frame
            sample_end = sample_start + self._num_frame
        imgs = imgs[sample_start:sample_end]
        bvp = nk.ppg_clean(bvp[sample_start:sample_end], self._fs, method='elgendi')
        bvp = (bvp - bvp.mean()) / bvp.std()

        return np.transpose(imgs, (3, 0, 1, 2)).astype('float32'), bvp.astype('float32')

    @property
    def fs(self):
        return self._fs

    @property
    def multiple(self):
        return self._multiple

class RGBNIRDatasetBase(Dataset):
    def __init__(self, train_list: List[Tuple[str, str, str]], test_list: List[Tuple[str, str, str]],
                 num_frame: int, fs: int, multiple: int, is_train: bool):
        self._num_frame = num_frame
        self._fs = fs
        self._multiple = multiple
        self._is_train = is_train
        self._train_list = train_list
        self._test_list = test_list
        self._id2range = list()
        self._num_clip_list = list()
        self._id2sampleid = dict()
        if self._is_train:
            for i in range(len(train_list)):
                length = np.min(
                    [np.load(train_list[i][0]).shape[0], np.load(train_list[i][1]).shape[0], np.load(train_list[i][2]).shape[0]]) - 60 - self._num_frame
                if length<=0:
                    continue
                length_avg = length // self._multiple
                start = 30 #AF lag
                end = length_avg
                while end <= length:
                    if end + length_avg > length:
                        end = length
                    self._id2range.append((i, start, end))
                    start += length_avg
                    end += length_avg
        if not self._is_train:
            for i in range(len(test_list)):
                num_clip = (np.min([np.load(test_list[i][0]).shape[0], np.load(test_list[i][1]).shape[0], np.load(test_list[i][2]).shape[0]])-60) // num_frame
                self._num_clip_list.append(num_clip)
            for i in range(sum(self._num_clip_list)):
                current_clips = 0
                for j in range(len(self._num_clip_list)):
                    next_clips = current_clips + self._num_clip_list[j]
                    if i < next_clips:
                        self._id2sampleid[i] = (j, i - current_clips)
                        break
                    else:
                        current_clips = next_clips

    def __len__(self):
        if self._is_train:
            return len(self._id2range)
        else:
            return sum(self._num_clip_list)

    def __getitem__(self, i: int):
        work_list = self._train_list if self._is_train else self._test_list
        if self._is_train:
            sampleid, start, end = self._id2range[i]
            imgs, imgs2, bvp = np.load(work_list[sampleid][0],mmap_mode='r'), np.load(work_list[sampleid][1],mmap_mode='r'), np.load(work_list[sampleid][2])
            num_frame, height, width, num_channel = imgs.shape

            sample_start = np.random.choice(np.arange(start, end))
            sample_end = sample_start + self._num_frame
        else:
            sampleid, innerid = self._id2sampleid[i]
            imgs, imgs2, bvp = np.load(work_list[sampleid][0],mmap_mode='r'), np.load(work_list[sampleid][1],mmap_mode='r'), np.load(work_list[sampleid][2])
            num_frame, height, width, num_channel = imgs.shape
            assert height == width
            sample_start = innerid * self._num_frame +30 #AF lag
            sample_end = sample_start + self._num_frame
        imgs, imgs2 = imgs[sample_start:sample_end], imgs2[sample_start:sample_end]
        bvp = nk.ppg_clean(bvp[sample_start:sample_end], self._fs, method='elgendi')
        bvp = (bvp - bvp.mean()) / bvp.std()

        return np.transpose(imgs, (3, 0, 1, 2)).astype('float32'), np.transpose(imgs2, (3, 0, 1, 2)).astype('float32'), bvp.astype('float32')

    @property
    def fs(self):
        return self._fs

    @property
    def multiple(self):
        return self._multiple

class MMSEDataset(RGBDatasetBase):
    def __init__(self,source_folder:str,fold_index:int,
                 num_frame:int,is_train:bool):
        assert fold_index<=4
        f_sub_list = list(range(5, 20)) + list(range(21, 28))
        m_sub_list = list(range(1, 18))
        self.workDir=source_folder
        self.sublist=['F%03d' %n for n in f_sub_list]+['M%03d' %n for n in m_sub_list]
        self.extra1=['M010','M011','F013','F014','F015','F016','F017','F018','F022']
        self.extra8=['M010','F013','F014','F015','F016','F017','F018','F022']
        self.extra9=['F022']
        self.extra14=['F022']
        trainNpyList = [];testNpyList = []
        for i in range(len(self.sublist)):
            if i>=fold_index*8 and i<(fold_index+1)*8:
                testNpyList=testNpyList+self.sub2videolist(self.sublist[i])
            else:
                trainNpyList=trainNpyList+self.sub2videolist(self.sublist[i])
        super().__init__(trainNpyList,testNpyList,num_frame,
                         25,2,is_train)

    def sub2videolist(self,sub:str):
        videolist=[]
        if os.path.isfile(self.workDir+sub+'T10_imgs.npy') and os.path.isfile(self.workDir+sub+'T10_bvp.npy') \
            and os.path.isfile(self.workDir+sub+'T11_imgs.npy') and os.path.isfile(self.workDir+sub+'T11_bvp.npy'):
            videolist.append((self.workDir+sub+'T10_imgs.npy',self.workDir+sub+'T10_bvp.npy'))
            videolist.append((self.workDir+sub+'T11_imgs.npy',self.workDir+sub+'T11_bvp.npy'))
        else:
            raise(ValueError('invalid file'))
        if str in self.extra1:
            if os.path.isfile(self.workDir + sub+'T1_imgs.npy') and os.path.isfile(self.workDir + sub+'T1_bvp.npy'):
                videolist.append((self.workDir + sub+'T1_imgs.npy', self.workDir + sub+'T1_bvp.npy'))
            else:
                raise(ValueError('invalid file'))
        if str in self.extra8:
            if os.path.isfile(self.workDir + sub+'T8_imgs.npy') and os.path.isfile(self.workDir + sub+'T8_bvp.npy'):
                videolist.append((self.workDir + sub+'T8_imgs.npy', self.workDir + sub+'T8_bvp.npy'))
            else:
                raise(ValueError('invalid file'))
        if str in self.extra9:
            if os.path.isfile(self.workDir + sub+'T9_imgs.npy') and os.path.isfile(self.workDir + sub+'T9_bvp.npy'):
                videolist.append((self.workDir + sub+'T9_imgs.npy', self.workDir + sub+'T9_bvp.npy'))
            else:
                raise(ValueError('invalid file'))
        if str in self.extra14:
            if os.path.isfile(self.workDir + sub+'T14_imgs.npy') and os.path.isfile(self.workDir + sub+'T14_bvp.npy'):
                videolist.append((self.workDir + sub+'T14_imgs.npy', self.workDir + sub+'T14_bvp.npy'))
            else:
                raise(ValueError('invalid file'))
        return videolist

