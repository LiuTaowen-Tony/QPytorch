import argparse
import torch
from torch import nn
from sr_experiments.network import PreResNet
from utils import make_quant_func, Id
from data import MyDataModule
import lightning
import numpy as np

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

def make_version_name(args):
    return (
        f"w{args.weight_bw}{args.weight_round[0]}"
        f"e{args.error_bw}{args.error_round[0]}"
        f"g{args.gradient_bw}{args.gradient_round[0]}"
        f"a{args.activation_bw}{args.activation_round[0]}"
        f"b{args.batch_size}"
    )


def load_state_dict_no_wrapper(model, state_dict):
    own_state = model.state_dict()
    for name in own_state.keys():
        name_with_prefix = "backbone." + name
        own_state[name].copy_(state_dict[name_with_prefix])
device = "cuda" if torch.cuda.is_available() else "cpu"

def test(args):
    full_precision_model = PreResNet(lambda: Id())
    stdt = torch.load(args.checkpoint_path, map_location=device)["state_dict"]
    load_state_dict_no_wrapper(full_precision_model, stdt)
    quant_funcs = make_quant_func(args)
    target_model = PreResNet(quant_funcs["make_ae_quant"])
    load_state_dict_no_wrapper(target_model, stdt)
    target_model.clean()
    full_precision_model.clean()
    data_loader = MyDataModule(args.batch_size)
    large_batch_data_loader = MyDataModule(1024)
    for x, y in data_loader.train_dataloader(): break
    for xl, yl in large_batch_data_loader.train_dataloader(): break

    # full precision model
    full_precision_model.clean()
    out = full_precision_model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    small_real_grads = full_precision_model.probe_grad()

    # low precision model
    target_model.clean()
    out = target_model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    small_quant_grads = target_model.probe_grad()

    # large batch full precision model
    full_precision_model.clean()
    out = full_precision_model(xl)
    loss = nn.CrossEntropyLoss()(out, yl)
    loss.backward()
    large_real_grads = full_precision_model.probe_grad()

    diff_small_quant_large_real_grads = {n : (small_quant_grads[n] - large_real_grads[n]) for n in small_quant_grads.keys()}
    diff_small_quant_small_real_grads = {n : (small_quant_grads[n] - small_real_grads[n]) for n in small_quant_grads.keys()}
    diff_small_real_large_real_grads = {n : (small_real_grads[n] - large_real_grads[n]) for n in small_quant_grads.keys()}

    tensorboard = lightning.pytorch.loggers.TensorBoardLogger("probe_check_point", name=make_version_name(args))

    for i, n in enumerate(small_quant_grads.keys()):
        if i % 4 != 0: continue
        tensorboard.experiment.add_histogram("diff_small_quant_large_real_grads_" + n, diff_small_quant_large_real_grads[n])
        tensorboard.experiment.add_histogram("diff_small_quant_small_real_grads_" + n, diff_small_quant_small_real_grads[n])
        tensorboard.experiment.add_histogram("diff_small_real_large_real_grads_" + n, diff_small_real_large_real_grads[n])
        tensorboard.experiment.add_histogram("real_grads" + n, large_real_grads[n])

        def save_histogram(data, file_prefix, index, args):
            hists, bins = np.histogram(data.cpu().numpy(), bins=100)
            with open(f"probe_check_point/{make_version_name(args)}/{file_prefix}_{index}_bins.npy", "wb") as f:
                np.save(f, bins)
            with open(f"probe_check_point/{make_version_name(args)}/{file_prefix}_{index}_hists.npy", "wb") as f:
                np.save(f, hists)
        
        save_histogram(diff_small_quant_large_real_grads[n], "diff_small_quant_large_real_grads", n, args)
        save_histogram(diff_small_quant_small_real_grads[n], "diff_small_quant_small_real_grads", n, args)
        save_histogram(diff_small_real_large_real_grads[n], "diff_small_real_large_real_grads", n, args)
        save_histogram(large_real_grads[n], "large_real_grads", i, args)

class NameSpace:pass
    
def main():
    args = NameSpace()
    for batch_size in [4, 8, 16, 32, 64, 128]:
        for precision in [1, 2, 4, 8]:
            for rounding in ["nearest", "stochastic"]:
                args.batch_size = batch_size
                args.weight_bw = precision
                args.error_bw = precision
                args.gradient_bw = precision
                args.activation_bw = precision
                args.weight_round = rounding
                args.error_round = rounding
                args.gradient_round = rounding
                args.activation_round = rounding
                args.checkpoint_path = f"full_precision_train/full_precision_reference/version_2/checkpoints/epoch=19-step=15640.ckpt"
                test(args)
                print("here")

if __name__ == "__main__":
    main()


# assumption fp32 is accurate
# single iteration dynamics
# assume high precision accumulation in matrix multiply
# question 1:
# is single iteration batched gradient estimation biased / comparing to full precision small batch / full precision full batch?
# 
# question 3:
# where does increasing batch size stop working?
# try batch size 4 8 16 32 64 128
# for full precision model
#
# question 2:
# can large batch trade off for error in small batch gradient estimation
# try batch size 4 8 16 32 64 128
# try 1 2 4 8 bits
# on low precision model and full precision model
# I guess not, because 
# 
#
# question 4:
# can large batch trade off for error in small batch gradient estimation if using stochastic rounding
# try batch size 4 8 16 32 64 128
# try 1 2 4 8 bits
# on low precision model and full precision model
# I guess not, because 
#
# question 5:
# what are the value range in gradient estimation?
# also how does these value range change over iterations? try epoch 0 5 10 20 30
# activation
# weight
# error
# gradient
#
# multiple iteration dynamics
# can I 
#
