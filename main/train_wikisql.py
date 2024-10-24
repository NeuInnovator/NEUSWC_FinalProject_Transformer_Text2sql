import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import Adafactor
from torch.utils.data import DataLoader
from monitor import LossMonitor
from model import Txt2SqlTransformer
from wikisql_dataset import get_dataset, mycollate
import config
class LitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        vocab_size: int = config.VOCAB_SIZE_WIKISQL,
        embed_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 16,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.model = Txt2SqlTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            d_model=d_model,
            num_layers=num_layers,
            n_head=n_heads,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
        self.lr = lr
        # 绘图帮助代码
        # self.train_losses = []  # 用于存储训练集的损失
        # self.val_losses = []  # 用于存储验证集的损失

    def forward(
        self,
        src: torch.Tensor,
        trg_input: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        trg_padding_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            src,
            trg_input,
            src_mask,
            trg_mask,
            src_padding_mask,
            trg_padding_mask,
            memory_mask,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        # 调用了forward函数
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        src, trg = batch
        trg_input = trg[:-1, :]
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.model.create_mask(
            src, trg_input
        )
        logits = self.model(
            src,
            trg_input,
            src_mask.to(self.device),
            trg_mask.to(self.device),
            src_padding_mask.to(self.device),
            trg_padding_mask.to(self.device),
            src_padding_mask.to(self.device),
        )
        trg_out = trg[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adafactor(
            self.model.parameters(),
            lr=self.lr,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False,
        )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
            collate_fn=mycollate,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            eval_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            collate_fn=mycollate,
        )
        return val_loader

if __name__ == "__main__":
    # 获取数据集
    train_ds, eval_ds = get_dataset()

    model = LitModel()
    # swa_epoch_start:在第几轮之后启动swa
    # swa_lrs:swa使用的学习率
    # SWA是随机权重平均:对于模型的参数权重进行平均，防止过拟合，增加泛化能力
    swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=5e-5,device="cuda")
    # 在验证集最小时进行保存模型，防止过拟合
    checkpointer_val_loss_min = pl.callbacks.ModelCheckpoint(
        dirpath="../model/wikisql",
        save_last=True,
        every_n_epochs=10,
    )
    checkpointer_train_loss_min = pl.callbacks.ModelCheckpoint(
        dirpath="./checkpoint/",
        monitor="train_loss",
        mode="min",
        filename="var=train_loss-epoch={epoch}-loss={train_loss:.4f}",  # 自定义命名格式
        auto_insert_metric_name=False,
        save_top_k=1,
    )
    logger = TensorBoardLogger("logs", name="text2sql")
    # 定义trainer
    # accumulate_grad_batches:每训练5轮进行一次反向传播
    find_lr_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[swa],
        logger=logger,
        accumulate_grad_batches=5,
        max_epochs=config.MAX_EPOCHS,
    )
    tuner = pl.tuner.Tuner(find_lr_trainer)  # 初始化 tuner
    lr_finder = tuner.lr_find(model)  # 查找学习率
    new_lr = lr_finder.suggestion()
    model.lr = new_lr*5
    print("---使用学习率：", new_lr)
    print('----------------------------开始训练----------------------------')
    loss_monitor = LossMonitor()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[ swa, checkpointer_train_loss_min,checkpointer_val_loss_min,loss_monitor],
        logger=logger,
        accumulate_grad_batches=5,
        max_epochs=config.MAX_EPOCHS,
    )
    trainer.fit(model)
    loss_monitor.plot_losses()
    print('----------------------------训练结束----------------------------')