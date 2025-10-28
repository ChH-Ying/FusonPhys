from models import *
from engine import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch import optim
import torch
import os
import numpy as np
import random

ex=Experiment('eval',save_git_info=False)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
else:
    raise(ValueError('cuda is not available'))

@ex.config
def my_config():
    result_dir = ""
    dataset_source = ""
    total_epoch=10
    selected_epoch=0
    lr=1e-4
    fold_id=0
    total_frame=896
    batch_size=2
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, seed,result_dir,dataset_source,total_epoch,selected_epoch,lr,fold_id,total_frame,batch_size):
    exp_dir = result_dir + '/%d/' % (int(_run._id))
    setup_deterministic(seed)

    dataset_params=[dataset_source,total_frame] if fold_id==-1 else [dataset_source,fold_id,total_frame]
    criterion=Neg_Pearson()
    model= FusionPhys.camera_only(3).to(device)
    engine=MMSEEngine(seed,device,exp_dir,model,dataset_params,optim.AdamW,total_epoch,lr, criterion,batch_size)
    engine.reset_for_evaluation()
    print(engine.eval(result_dir+f'weights_e{selected_epoch}.pth'))

def setup_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)


