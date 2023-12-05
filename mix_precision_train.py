from os import path
from typing import Optional

from data import MyDataModule

import torch
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

from sr_experiments.network import PreResNet
from torch import nn
from qtorch.number import FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from torch.utils.data import DataLoader

import torch.nn.functional as F
import argparse
import lightning as L
from utils import make_quant_func, Id


from lightning.pytorch.loggers import TensorBoardLogger


rounding_options_choices = ["nearest", "stochastic", ]

parser = argparse.ArgumentParser(description=' Example')
parser.add_argument('-b', "--batch-size", type=int,
                    default=128, help='batch size')
parser.add_argument('-l', "--learning-rate", type=float,
                    default=0.1, help='learning rate')
parser.add_argument('-m', "--momentum", type=float, default=0, help='momentum')
parser.add_argument("--loss-scale", type=int, default=1,
                    help='loss scaling factor, 1 for no scaling')
parser.add_argument("--weight-ew", type=int, default=3,
                    help='exponent bit width for weight')
parser.add_argument("--error-ew", type=int, default=3,
                    help='exponent bit width for error')
parser.add_argument("--gradient-ew", type=int, default=3,
                    help='exponent bit width for gradient')
parser.add_argument("--activation-ew", type=int, default=3,
                    help='exponent bit width for activation')
parser.add_argument('-w', "--weight-bw", type=int, default=3,
                    help='mantissa bit width for weight')
parser.add_argument('-e', "--error-bw", type=int, default=3,
                    help='mantissa bit width for error')
parser.add_argument('-g', "--gradient-bw", type=int,
                    default=3, help='mantissa bit width for gradient')
parser.add_argument('-a', "--activation-bw", type=int,
                    default=3, help='mantissa bit width for activation')
parser.add_argument("--weight-round", default="nearest",
                    choices=rounding_options_choices, help='weight rounding method')
parser.add_argument("--error-round",  default="nearest",
                    choices=rounding_options_choices, help='error rounding method')
parser.add_argument("--gradient-round", default="nearest",
                    choices=rounding_options_choices, help='gradient rounding method')
parser.add_argument("--activation-round", default="nearest",
                    choices=rounding_options_choices, help='activation rounding method')
parser.add_argument("--checkpoint-path", default=None, help='checkpoint path')
parser.add_argument("--mix-precision", type=lambda x: x ==
                    "True", default=True, help='is mix precision train')

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()


class LitClassifier(LightningModule):
    def _apply_model_weights(self, model, quant_func):
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = quant_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data = quant_func(m.bias.data)

    def __init__(self, args):
        super().__init__()
        def make_ae_quant(): return Id()
        self.args = args
        quant_funcs = make_quant_func(args)
        self.weight_quant = quant_funcs["weight_quant"]
        self.grad_quant = quant_funcs["grad_quant"]
        make_ae_quant = quant_funcs["make_ae_quant"]
        self.backbone = PreResNet(make_ae_quant)
        self.reference_model = PreResNet(lambda: Id())
        if args.checkpoint_path:
            ckpt = torch.load(args.checkpoint_path)
            self.load_state_dict(ckpt["state_dict"])
        self._apply_model_weights(self.backbone, self.weight_quant)
        # self.backbone = torch.jit.trace(self.backbone, torch.rand(1, 3, 32, 32))
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["backbone"])

    def _make_float(self, mantissa_bit_width: Optional[int] = None):
        return FloatingPoint(exp=8, man=mantissa_bit_width)

    def forward(self, x):
        # use forward for inference/predictions
        return self.backbone(x)

    def copy_model_weights_to_reference_weights(self):
        for (name, param), (ref_name, ref_param) in zip(
            self.backbone.named_parameters(),
                self.reference_model.named_parameters()):
            ref_param.data = param.data.clone()

    def log_dict_hist(self, dict, prefix=""):
        tensorboard = self.logger.experiment
        for name, tensor in dict.items():
            tensorboard.add_histogram(
                prefix + name, tensor, self.current_epoch)

    def opt_step(self, loss, opt):
        opt.step()

    def model_grads_to_master_grads(self):
        for model, master in zip(self.backbone.parameters(), self.reference_model.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(self.grad_quant(model.grad.data))

    def master_grad_apply(self, fn):
        for master in (self.reference_model.parameters()):
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data = fn(master.grad.data)

    def master_params_to_model_params(self):
        for model, master in zip(self.backbone.parameters(), self.reference_model.parameters()):
            model.data.copy_(self.weight_quant(master.data))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        loss = loss * args.loss_scale
        opt = self.optimizers()
        self.backbone.zero_grad()
        self.manual_backward(loss)
        if args.mix_precision:
            self.model_grads_to_master_grads()
        self.master_grad_apply(lambda x: x / args.loss_scale)
        opt.step()
        if args.mix_precision:
            self.master_params_to_model_params()
        self._apply_model_weights(self.backbone, self.weight_quant)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("valid_loss", loss, on_step=True)
        self.log_dict({"valid_loss": loss, "valid_acc": acc, },
                      on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"test_loss": loss, "test_acc": acc, },
                      on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = SGD(self.parameters(),
                        lr=self.args.learning_rate,
                        momentum=self.args.momentum)
        return optimizer


def make_version_name(args):
    return (
        ("mix_precision_" if args.mix_precision else "no_mix_") +
        f"_ls{args.loss_scale}_"
        f"w{args.weight_bw}{args.weight_ew}{args.weight_round[0]}"
        f"e{args.error_bw}{args.error_ew}{args.error_round[0]}"
        f"g{args.gradient_bw}{args.gradient_ew}{args.gradient_round[0]}"
        f"a{args.activation_bw}{args.activation_ew}{args.activation_round[0]}"
        f"lr{args.learning_rate}"
        f"b{args.batch_size}"
    )


def cli_main():
    args = parser.parse_args()
    model = LitClassifier(args)
    print(args)
    logger = TensorBoardLogger(
        "mix_precision_test", name=make_version_name(args), )
    trainer = L.Trainer(accelerator="gpu", max_epochs=100,
                        logger=logger, enable_progress_bar=True)
    datamodule = MyDataModule(args.batch_size)
    # trainer.test(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    # predictions = trainer.predict(ckpt_path="best", datamodule=datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_main()
