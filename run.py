import os
import copy
import shutil
import pytorch_lightning as pl

from pytorch_lightning.strategies import DDPStrategy 

from clip.config import ex
from clip.modules import CLIPransformerSS
from clip.datamodules.multitask_datamodule import MTDataModule

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = CLIPransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    
    # Checkpoint Callback 
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    # Logger
    logger = pl.loggers.CSVLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )

    target_dir = './result_model_files' + '/' + _config["log_dir"] +'/' + f'{exp_name}_seed{_config["seed"]}' + '/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)
    else:
        os.makedirs(target_dir)
    shutil.copy2('./clip/modules/clip_meta_prompt_module.py', target_dir)
    shutil.copy2('./clip/modules/vision_transformer_prompts.py', target_dir)
    shutil.copy2('./clip/config.py', target_dir)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
    
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1 
    

    # --- PyTorch Lightning 2.0  ---
    
    ddp_strategy = DDPStrategy(find_unused_parameters=True) 
    

    trainer = pl.Trainer(
        devices=_config["num_gpus"],       
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",                 
        # strategy="ddp",                   
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps == -1 else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        # resume_from_checkpoint 
        # weights_summary="top" 
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # replace_sampler_ddp 
        # prepare_data_per_node 
        # flush_logs_every_n_steps 
    )

    if not _config["test_only"]:
        resume_ckpt = _config["resume_from"] if _config["resume_from"] is not None else None
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
    else:
        trainer.test(model, datamodule=dm)



# CUDA_VISIBLE_DEVICES=0 python run.py with data_root=/disk1/longw/Datasets/mmimdb/ num_gpus=1 num_nodes=1 per_gpu_batchsize=128 task_finetune_mmimdb exp_name=exp_mmimdb
# CUDA_VISIBLE_DEVICES=7 python run.py with data_root=/disk1/longw/Datasets/mmimdb/ num_gpus=1 num_nodes=1 per_gpu_batchsize=64 task_finetune_mmimdb load_path=/disk1/longw/work/MissingModality/CPB2/result/exp_mmimdb_seed0/version_6/checkpoints/epoch=17-step=2284.ckpt test_only=True test_ratio=0.9 test_type=image exp_name=exp_test

