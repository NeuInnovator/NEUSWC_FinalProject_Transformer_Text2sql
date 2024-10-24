import os
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from pytorch_lightning import Callback

class LossMonitor(Callback):
    def __init__(self, save_dir="./img"):
        super().__init__()
        self.train_losses = []  # 存储每个step的训练损失
        self.val_losses = []  # 存储每个step的验证损失
        self.epochs = []  # 存储每个epoch的编号
        self.save_dir = save_dir  # 保存图片的目录

        # 创建保存目录（如果不存在）
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """在每个训练step结束时记录损失。"""
        loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """在每个验证step结束时记录损失。"""
        loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
        self.val_losses.append(loss)

    def on_train_epoch_end(self, trainer, pl_module):
        """在每个epoch结束时记录epoch编号并输出平均训练损失。"""
        epoch = trainer.current_epoch
        self.epochs.append(epoch)
        avg_train_loss = trainer.callback_metrics["train_loss"].item()
        print(f"Epoch {epoch} - Avg Train Loss: {avg_train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """在每个epoch结束时输出平均验证损失。"""
        epoch = trainer.current_epoch
        avg_val_loss = trainer.callback_metrics["val_loss"].item()
        print(f"Epoch {epoch} - Avg Val Loss: {avg_val_loss}")

    def plot_losses(self):
        """绘制损失图像并保存。"""
        train_len = len(self.train_losses)
        val_len = len(self.val_losses)
        epoch_num = len(self.epochs)

        steps_per_train_epoch = train_len // epoch_num
        steps_per_val_epoch = val_len // epoch_num

        # 绘制训练损失的折线图
        plt.figure(figsize=(15, 10))
        plt.plot(self.train_losses, label="Training Loss")
        epoch_ticks = np.arange(0, train_len+1, steps_per_train_epoch*5)
        plt.xticks(epoch_ticks, [str(e*5) for e in range(len(epoch_ticks))])
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.grid(axis='y')
        plt.legend()
        self.save_plot('train_epochs_loss')

        # 绘制验证损失的折线图
        plt.figure(figsize=(15, 10))
        plt.plot(self.val_losses, label="Validation Loss")
        epoch_ticks = np.arange(0, val_len+1, steps_per_val_epoch*5)
        plt.xticks(epoch_ticks, [str(e*5) for e in range(len(epoch_ticks))])
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.grid(axis='y')
        plt.legend()
        self.save_plot('val_epochs_loss')
        # 同时绘制训练和验证损失的对比图

        # 创建训练和验证的 step 数组
        train_steps = np.arange(train_len)
        val_steps = np.arange(val_len)
        # 绘制训练损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, self.train_losses, label="Training Loss")
        # 绘制验证损失曲线
        plt.plot(val_steps*train_len/val_len, self.val_losses, label="Validation Loss")
        # 设置 x 轴刻度：每个 epoch 的位置
        epoch_ticks = np.arange(0, max(train_len, val_len) + 1,
                                max(steps_per_train_epoch, steps_per_val_epoch)*5)
        plt.xticks(epoch_ticks, [str(e*5) for e in range(len(epoch_ticks))])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(axis='y')
        plt.legend()
        self.save_plot('all_epochs_loss')

    def save_plot(self, plot_name):
        """使用当前时间和图名保存图片。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.save_dir, f"{plot_name}_{timestamp}.png")
        plt.savefig(file_path)
        print(f"{plot_name} plot saved at: {file_path}")


# # 创建 LossMonitor 实例
# loss_monitor = LossMonitor()
#
# # 模拟 10 个 epoch，每个 epoch 包含 20 个训练和验证 step 的损失
# epochs = 100
# steps_per_epoch = 20
#
# # 生成一些模拟的损失数据
# train_losses = np.random.uniform(0.2, 1.0, size=epochs * steps_per_epoch).tolist()
# val_losses = np.random.uniform(0.2, 1.0, size=epochs * 10).tolist()
#
# # 将生成的损失数据填充到 LossMonitor 实例中
# loss_monitor.train_losses = train_losses
# loss_monitor.val_losses = val_losses
# loss_monitor.epochs = list(range(epochs))
#
# # 测试 plot_losses() 函数
# loss_monitor.plot_losses()