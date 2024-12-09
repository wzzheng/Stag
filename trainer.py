import argparse, os, sys, datetime
from omegaconf import OmegaConf, DictConfig
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.diffusion_utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints
from main.dataset import *
from main.du3d3d import Du3d3d
from configs.train_config import get_parser
from mmdet3d.datasets import build_dataset



torch.multiprocessing.set_sharing_strategy('file_system')

def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))




if __name__ == "__main__":
    
    #global du3d3d3d
    
    global_number01 = 10
    global_number02 = 10
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    print(now)
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    
    ## Extends existing argparse by default Trainer attributes
    parser = Trainer.add_argparse_args(parser)
    opts = parser.parse_args()#############################
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)
    print(args.base)
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
    
    
    ## register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
        
    num_nodes = 1
    ngpu_per_node = 1
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr
        
    #shangmiandouky!!!!!1
    
    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    print("config.nus.trainconfig.nus.trainconfig.nus.train",config.nus.train)
    print(DictConfig)
    train_dataset = build_dataset(
        OmegaConf.to_container(config.nus.train, resolve=True)
    )
    val_dataset = build_dataset(
        OmegaConf.to_container(config.nus.val, resolve=True)
    )

    collate_fn_param = {
        #"tokenizer": CLIPTokenizer.from_pretrained(config.modulemagic.pretrained_model_name_or_path, subfolder="tokenizer"),#self.tokenizer,
        "template": config.nus.template,
        "bbox_mode": config.modulemagic.bbox_mode,
        "bbox_view_shared": False,#config.modulemagic.bbox_view_shared,
        "bbox_drop_ratio": 0,#config.runner.bbox_drop_ratio,
        "bbox_add_ratio": 0,#config.runner.bbox_add_ratio,
        "bbox_add_num": 3,#config.runner.bbox_add_num,
        #"keyframe_rate": 6,#self.cfg.runner.keyframe_rate,
    }




    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True,
    collate_fn=partial(
        collate_fn, is_train=True, **collate_fn_param),   
    batch_size=config.data.params.batch_size,
    num_workers=config.data.params.num_workers, pin_memory=False,
    prefetch_factor=1,#self.cfg.runner.prefetch_factor,
    persistent_workers=True,)
    
    #print(train_dataset)
    data = train_dataloader
    print("data",data)
    '''
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"       
    '''
    
    
    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False    
    
    ## trainer config: others

    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    
    
    #################3
    #parser = get_parser()
    #opts = parser.parse_args()#############################
    #global du3d3d3d 
    #import main.d3d3d3d
    obj = Du3d3d(opts)#,model
    d3d3d3d.shared_obj = obj
    ################3
    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config['strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                ## deepspeed
                if trainer_kwargs['precision'] == 16:
                    #with torch.cuda.amp.autocast():
                    trainer.fit(model, data)
                else:
                    #with torch.cuda.amp.autocast():                
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>") ## this is default
                ## ddpsharded
                trainer.fit(model, data)
        except Exception:
            #melk()
            raise

    #print(a)