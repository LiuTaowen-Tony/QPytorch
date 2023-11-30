import os
from sr_experiments.network import PreResNet
from sr_experiments.data import get_loaders

from qtorch.number import FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm

import argparse

rounding_options_choices = ["nearest", "stochastic", ]

parser = argparse.ArgumentParser(description=' Example')
parser.add_argument('-b', "--batch-size", type=int, default=128, help='batch size')
parser.add_argument('-l', "--learning-rate", type=float, default=0.05, help='learning rate')
parser.add_argument('-m', "--momentum", type=float, default=0.9, help='momentum')
parser.add_argument('-w', "--weight-bw", type=int, default=3, help='mantissa bit width for weight')
parser.add_argument('-e', "--error-bw", type=int, default=3, help='mantissa bit width for error')
parser.add_argument('-g', "--gradient-bw", type=int, default=3, help='mantissa bit width for gradient')
parser.add_argument('-a', "--activation-bw", type=int, default=3, help='mantissa bit width for activation')
parser.add_argument("--weight-round", default="stochastic", choices=rounding_options_choices, help='weight rounding method')
parser.add_argument("--error-round",  default="stochastic",choices=rounding_options_choices, help='error rounding method')
parser.add_argument("--gradient-round", default="stochastic", choices=rounding_options_choices, help='gradient rounding method')
parser.add_argument("--activation-round", default="stochastic", choices=rounding_options_choices, help='activation rounding method')

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()

# define two floating point formats
exponent_width = 8
mans = [
    None, 
    FloatingPoint(exp=exponent_width, man=1), 
    FloatingPoint(exp=exponent_width, man=2),
    FloatingPoint(exp=exponent_width, man=3), 
    FloatingPoint(exp=exponent_width, man=4), 
    FloatingPoint(exp=exponent_width, man=5), 
    FloatingPoint(exp=exponent_width, man=6),
    FloatingPoint(exp=exponent_width, man=7)
]

def _apply_model_weights(model, quant_func):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data = quant_func(m.weight.data)
            if m.bias is not None:
                m.bias.data = quant_func(m.bias.data)


# def grad_quant(self, model, quant_func):
#     model_params = model.parameters()
#     for p in (model_params):
#         if p.grad is None:
#             p.grad = Variable(model.data.new(*model.data.size()))
#         p.grad.data = quant_func(p.grad.data)

def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="eval": model.eval()

    ttl = 0
    _apply_model_weights(model, weight_quant)
    with torch.autograd.set_grad_enabled(phase=="train"):
        # for i, (input, target) in enumerate(loader):
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)

            output = model(input)
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            print(pred.eq(target.data.view_as(pred)).sum() / len(input))
            ttl += input.size()[0]

            if phase=="train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
    }


def train_early_stop(loaders, model, loss_function, optimizer, epochs, patience):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    for epoch in range(epochs):
        result = run_epoch(loaders['train'], model, loss_function, optimizer=optimizer, phase="train")
        print("Epoch: {}, Train Loss: {}, Train Accuracy: {}".format(epoch, result['loss'], result['accuracy']))

        model.eval()
        val_loss = 0
        result = run_epoch(loaders['test'], model, loss_function, optimizer=optimizer, phase="eval")
        print("Epoch: {}, Test Loss: {}, Test Accuracy: {}".format(epoch, result['loss'], result['accuracy']))
        val_loss = result['loss']
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            early_stop = True
            break

    if not early_stop:
        print("Training completed without early stopping")

class Id(nn.Module):
    def forward(self, x):
        return x

if "__main__" == __name__:
    # make_ae_quant = lambda : Quantizer(forward_number=mans[args.activation_bw], backward_number=mans[args.error_bw],
    #                         forward_rounding=args.activation_round, backward_rounding=args.error_round)
    make_ae_quant = lambda : Id()
    weight_quant = quantizer(forward_number=mans[args.weight_bw],
                            forward_rounding=args.weight_round)
    grad_quant = quantizer(forward_number=mans[args.gradient_bw],
                            forward_rounding=args.gradient_round)
    model = PreResNet(make_ae_quant, num_classes=10, depth=20)
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    # optimizer = OptimLP(optimizer,
    #                     weight_quant=weight_quant,
    #                     grad_quant=grad_quant,
    #                     momentum_quant=grad_quant,
    # )
    loaders = get_loaders(args.batch_size)

    train_early_stop(loaders, model, F.cross_entropy, optimizer, 100, 10)