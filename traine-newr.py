
import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging

import pytorch_lightning as pl
import torch
from torch.optim import Adam
import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from configs.infer_config import get_parser
from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime
from utils.diffusion_utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints


class ViewCrafterLightningModule(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        #self.view_crafter = ViewCrafter(opts)
        self.pvd = ViewCrafter(opts)
        self.opts = opts
        self.loss_fn = torch.nn.MSELoss()  #

    def forward(self, x):
        return self.pvd.nvs_single_view()

    def training_step(self, batch, batch_idx):
        # 
        images, targets = batch
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, targets)  # 
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        lr = 0.0001
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
        
        
        
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir=None, transform=None, num_samples=100, image_size=(3, 64, 64)):
        self.data_dir = data_dir
        self.transform = transform
        self.num_samples = num_samples
        self.image_size = image_size
        #
        self.data = [self._generate_random_data() for _ in range(num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

    def _generate_random_data(self):
        # 
        image = torch.randn(*self.image_size)  # 
        target = torch.randint(0, 2, (1,)).item()
        return image, target

#
train_dataset = CustomDataset(num_samples=100, image_size=(3, 64, 64))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


from pytorch_lightning import Trainer

if __name__ == "__main__":

    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    #
    #prefix = datetime.now().strftime("%Y%m%d_%H%M")
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    #parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    
    #parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) 

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)

    ## load checkpoints
    model = load_checkpoints(model, config.model)
    print("99999999999999999999999999999999999999999")
    #
    trainer = Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0
    )
    print("555555555555555555555555555555")
    for name, param in model.named_parameters():
        print(name, param.shape)

    #
    trainer.fit(model, train_loader)
