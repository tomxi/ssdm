import ssdm
from ssdm import scanner_lightning as scl

tb_logger = scl.L.pytorch.loggers.TensorBoardLogger(save_dir="./runs/", name='slm')

trainer = scl.L.pytorch.Trainer(
    num_sanity_val_steps = -1,
    max_epochs = 100,
    devices=2,
    accelerator="gpu",
    accumulate_grad_batches=16,
    logger=tb_logger,
)

dm = scl.PairDataModule(ds_module=ssdm.slm, perf_margin=0.01)
lit_net = scl.LitUtilModel()

trainer.fit(lit_net, datamodule=dm)