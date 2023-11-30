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
parser.add_argument('-b', "--batch-size", type=int, default=128, help='batch size')
parser.add_argument('-l', "--learning-rate", type=float, default=0.1, help='learning rate')
parser.add_argument('-m', "--momentum", type=float, default=0, help='momentum')
parser.add_argument('-w', "--weight-bw", type=int, default=3, help='mantissa bit width for weight')
parser.add_argument('-e', "--error-bw", type=int, default=3, help='mantissa bit width for error')
parser.add_argument('-g', "--gradient-bw", type=int, default=3, help='mantissa bit width for gradient')
parser.add_argument('-a', "--activation-bw", type=int, default=3, help='mantissa bit width for activation')
parser.add_argument("--weight-round", default="nearest", choices=rounding_options_choices, help='weight rounding method')
parser.add_argument("--error-round",  default="nearest",choices=rounding_options_choices, help='error rounding method')
parser.add_argument("--gradient-round", default="nearest", choices=rounding_options_choices, help='gradient rounding method')
parser.add_argument("--activation-round", default="nearest", choices=rounding_options_choices, help='activation rounding method')
parser.add_argument("--checkpoint-path", default=None, help='checkpoint path')

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
        make_ae_quant = lambda : Id()
        self.args = args
        quant_funcs = make_quant_func(args)
        self.weight_quant = quant_funcs["weight_quant"]
        self.grad_quant = quant_funcs["grad_quant"]
        make_ae_quant = quant_funcs["make_ae_quant"]
        self.backbone = PreResNet(make_ae_quant)
        self.reference_model = PreResNet(lambda : Id())
        if args.checkpoint_path:
            ckpt = torch.load(args.checkpoint_path)
            self.load_state_dict(ckpt["state_dict"])
        self._apply_model_weights(self.backbone, self.weight_quant)
        # self.backbone = torch.jit.trace(self.backbone, torch.rand(1, 3, 32, 32))
        self.automatic_optimization=False
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


    def optimise_and_probe(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        self.copy_model_weights_to_reference_weights()
        self.backbone.clean()
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)

        self.reference_model.clean()
        ref_y_hat = self.reference_model(x)
        ref_loss = F.cross_entropy(ref_y_hat, y)

        acts = self.backbone.probe_activation() #dict of name to tensor
        weights = self.backbone.probe_weight()
        self.backbone.zero_grad()
        loss.backward()
        errs = self.backbone.probe_err()
        grads = self.backbone.probe_grad()

        ref_acts = self.reference_model.probe_activation()
        ref_weights = self.reference_model.probe_weight()
        self.reference_model.zero_grad()
        ref_loss.backward()
        ref_errs = self.reference_model.probe_err()
        ref_grads = self.reference_model.probe_grad()

        clip = lambda x: x.clamp(-1, 3)
        diff_acts = {n: (ra - acts[n]) for n, ra in ref_acts.items()}
        diff_weights = {n: rw - weights[n] for n, rw in ref_weights.items()}
        diff_errs = {n: re - errs[n] for n, re in ref_errs.items()}
        diff_grads = {n: rg - grads[n] for n, rg in ref_grads.items()}
        relative_diff_acts = {n: clip((ra - acts[n]) / acts[n]) for n, ra in ref_acts.items()}
        relative_diff_weights = {n: clip((rw - weights[n]) / weights[n]) for n, rw in ref_weights.items()}
        relative_diff_errs = {n: clip((re - errs[n]) / errs[n]) for n, re in ref_errs.items()}
        relative_diff_grads = {n: clip((rg - grads[n]) / grads[n]) for n, rg in ref_grads.items()}

        self.opt_step(loss, opt)
        new_weights = self.backbone.probe_weight()
        ref_optimiser = SGD(self.reference_model.parameters(),
                            lr=args.learning_rate,
                            momentum=args.momentum)
        ref_optimiser.step()
        ref_new_weights = self.reference_model.probe_weight()

        diff_new_weights = {n: rw - new_weights[n] for n, rw in ref_new_weights.items()}
        relative_diff_new_weights = {n: clip((rw - new_weights[n]) / new_weights[n]) for n, rw in ref_new_weights.items()}

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("ref_loss", ref_loss.item())
        self.log("real_loss", loss.item())

        self.log_dict_hist(diff_acts, "diff_acts")
        self.log_dict_hist(diff_weights, "diff_weights")
        self.log_dict_hist(diff_errs, "diff_errs")
        self.log_dict_hist(diff_grads, "diff_grads")
        self.log_dict_hist(diff_new_weights," diff_new_weights")

        self.log_dict_hist(relative_diff_acts," relative_diff_acts")
        self.log_dict_hist(relative_diff_weights," relative_diff_weights")
        self.log_dict_hist(relative_diff_errs," relative_diff_errs")
        self.log_dict_hist(relative_diff_grads," relative_diff_grads")
        self.log_dict_hist(relative_diff_new_weights," relative_diff_new_weights")

        self.log_dict_hist(acts," acts")
        self.log_dict_hist(weights," weights")
        self.log_dict_hist(errs," errs")
        self.log_dict_hist(grads," grads")
        self.log_dict_hist(new_weights," new_weights")
        self.log_dict_hist(ref_acts," ref_acts")
        self.log_dict_hist(ref_weights," ref_weights")
        self.log_dict_hist(ref_errs," ref_errs")
        self.log_dict_hist(ref_grads," ref_grads")
        self.log_dict_hist(ref_new_weights," ref_new_weights")
        return loss
        
    


    def log_dict_hist(self, dict, prefix = ""):
        tensorboard = self.logger.experiment
        for name, tensor in dict.items():
            tensorboard.add_histogram(prefix + name, tensor, self.current_epoch)


    def opt_step(self, loss, opt):
        opt.step()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            return self.optimise_and_probe(batch, batch_idx)
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("valid_loss", loss, on_step=True)
        self.log_dict({"valid_loss": loss, "valid_acc": acc, }, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"test_loss": loss, "test_acc": acc,}, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = SGD(self.parameters(), 
                        lr=self.args.learning_rate, 
                        momentum=self.args.momentum)
        optimizer = OptimLP(optimizer,
                            weight_quant=self.weight_quant,
                            grad_quant=self.grad_quant,
                            momentum_quant=self.grad_quant,
        )
        return optimizer




def make_version_name(args):
    return (
        f"w{args.weight_bw}{args.weight_round[0]}"
        f"e{args.error_bw}{args.error_round[0]}"
        f"g{args.gradient_bw}{args.gradient_round[0]}"
        f"a{args.activation_bw}{args.activation_round[0]}"
        f"lr{args.learning_rate}"
        f"b{args.batch_size}"
    )



def cli_main():
    args = parser.parse_args()
    model = LitClassifier(args)
    print(args)
    logger = TensorBoardLogger("tb_logs_new", name=make_version_name(args), )
    trainer = L.Trainer(accelerator="gpu", max_epochs=100, logger=logger, devices=1)
    datamodule = MyDataModule(args.batch_size)
    # trainer.test(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    predictions = trainer.predict(ckpt_path="best", datamodule=datamodule)
    print(predictions[0])


if __name__ == "__main__":
    cli_main()
