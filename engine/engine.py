import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Callable,Optional
import functools
from models import IrrelevantPowerRatio
from utils import *
import random

#For camera-radar fusion, please refer to the training code in "Equitable Plethysmography Blending Camera and 77 GHz Radar Sensing for Equitable, Robust Plethysmography"

class EngineBase():
    def __init__(self,seed,device,exp_dir,model,num_epoch,train_dataset,test_dataset,optimizer,scheduler,criterion,batch_size):
        self._seed=seed
        self._set_deterministic_state(self._seed)
        self._device=device
        self._exp_dir=exp_dir
        self._model=model
        self._num_epoch=num_epoch

        self._train_dataset=train_dataset
        self._test_dataset=test_dataset
        self._train_gen=torch.Generator().manual_seed(seed)
        self._test_gen=torch.Generator().manual_seed(seed+1)
        self._train_dataloader=DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=False,generator=self._train_gen)
        self._test_dataloader=DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,drop_last=False,generator=self._test_gen)
        self._optimizer=optimizer
        self._scheduler=scheduler
        self._criterion=criterion
        self._ipr=IrrelevantPowerRatio(Fs=self._train_dataset.fs, high_pass=40, low_pass=250)

    def _set_deterministic_state(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        if self._model.is_fusion:
            self._train_fusion()
        else:
            self._train_single()

    def eval(self,weights:Optional[str]=None):
        if self._model.is_fusion:
            return self._eval_fusion(weights)
        else:
            return self._eval_single(weights)

    def _train_single(self):
        min_ipr=1
        min_ipr_e=-1
        for e in range(self._num_epoch):
            epoch_seed = self._seed + e
            self._set_deterministic_state(epoch_seed)
            train_ipr=0
            self._model.train()
            for x,gt in self._train_dataloader:
                x=x.to(self._device)
                gt=gt.to(self._device)
                pred=self._model(x,None)
                loss=self._criterion(pred,gt)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                ipr = torch.mean(self._ipr(pred.clone().detach()))
                train_ipr += ipr.item()
            train_ipr/=len(self._train_dataset)
            if train_ipr<min_ipr:
                min_ipr=train_ipr
                min_ipr_e=e
            print(f"ipr_epoch{e}: {train_ipr}")
            torch.save(self._model.state_dict(), self._exp_dir+f"weights_e{e}.pth")
            self._scheduler.step()
        print(min_ipr_e,min_ipr)


    def _train_fusion(self):
        min_ipr=1
        min_ipr_e=-1
        for e in range(self._num_epoch):
            epoch_seed = self._seed + e
            self._set_deterministic_state(epoch_seed)
            train_ipr = 0
            self._model.train()
            for x1,x2,gt in self._train_dataloader:
                x1=x1.to(self._device)
                x2=x2.to(self._device)
                gt=gt.to(self._device)
                pred=self._model(x1,x2)
                loss=self._criterion(pred,gt)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                ipr = torch.mean(self._ipr(pred.clone().detach()))
                train_ipr += ipr.item()
            train_ipr /= len(self._train_dataset)
            if train_ipr<min_ipr:
                min_ipr=train_ipr
                min_ipr_e=e
            print(f"ipr_epoch{e}: {train_ipr}")

            torch.save(self._model.state_dict(), self._exp_dir+f"weights_e{e}.pth")
            self._scheduler.step()
        print(min_ipr_e, min_ipr)

    def _eval_single(self,weights:Optional[str]=None):
        self._set_deterministic_state(self._seed)
        if weights is not None:
            self._model.load_state_dict(torch.load(weights, map_location=self._device))
        self._model.eval()
        HR_pred=list()
        HR_gt=list()
        with torch.no_grad():
            for x,gt in self._test_dataloader:
                x = x.to(self._device)
                pred = self._model(x,None)
                pred = pred.detach().cpu().numpy()
                gt = gt.numpy()
                for b in range(pred.shape[0]):
                    HR_pred.append(get_HR(pred[b],self._test_dataset.fs))
                    HR_gt.append(get_HR(gt[b], self._test_dataset.fs))
        return metrics(np.array(HR_pred),np.array(HR_gt))

    def _eval_fusion(self,weights:Optional[str]=None):
        self._set_deterministic_state(self._seed)
        if weights is not None:
            self._model.load_state_dict(torch.load(weights, map_location=self._device))
        self._model.eval()
        HR_pred=list()
        HR_gt=list()
        with torch.no_grad():
            for x1,x2,gt in self._test_dataloader:
                x1 = x1.to(self._device)
                x2 = x2.to(self._device)
                pred = self._model(x1,x2)
                pred = pred.detach().cpu().numpy()
                gt = gt.numpy()
                for b in range(pred.shape[0]):
                    HR_pred.append(get_HR(pred[b],self._test_dataset.fs))
                    HR_gt.append(get_HR(gt[b], self._test_dataset.fs))
        return metrics(np.array(HR_pred),np.array(HR_gt))

    def reset_for_evaluation(self):
        self._set_deterministic_state(self._seed)
        self._test_generator = torch.Generator().manual_seed(self._seed + 1)
        self._test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self._test_dataloader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            generator=self._test_generator
        )

class MMSEEngine(EngineBase):
    def __init__(self,seed,device,exp_dir,model,dataset_params,optimizer:Callable[...,optim.Optimizer],num_epoch=10,lr=1e-4,criterion=None,batch_size=2):
        _train_dataset=MMSEDataset(*dataset_params,is_train=True)
        _test_dataset=MMSEDataset(*dataset_params,is_train=False)
        _optimizer = optimizer(self._model.parameters(), lr=lr)
        _scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self._optimizer, T_0=10,
                                                                               T_mult=1,
                                                                               eta_min=1e-8)
        _num_epoch=(num_epoch+_train_dataset.multiple-1)//_train_dataset.multiple
        super().__init__(seed,device,exp_dir,model,_num_epoch,_train_dataset,_test_dataset,_optimizer,_scheduler,criterion,batch_size)



