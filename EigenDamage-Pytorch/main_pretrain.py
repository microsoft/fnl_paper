'''Train CIFAR10/CIFAR100 with PyTorch.'''
from __future__ import print_function
import json
import math
import os
import pdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from utils.common_utils import PresetLRScheduler, makedirs

try:
    from frob import FactorizedConv, frobdecay, frobenius_norm, non_orthogonality, patch_module
    from make import compress_model, parameter_count
except ImportError:
    print("Failed to import factorization")

# fetch args
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--epoch', default=150, type=int)
parser.add_argument('--decay_every', default=60, type=int)
parser.add_argument('--decay_ratio', default=0.1, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)
parser.add_argument('--rank-scale', default=0.0, type=float)
parser.add_argument('--wd2fd', action='store_true')
parser.add_argument('--spectral', action='store_true')
parser.add_argument('--kaiming', action='store_true')
parser.add_argument('--target-ratio', default=0.0, type=float)
parser.add_argument('--auto-resume', action='store_true')
args = parser.parse_args()

# init model
net = get_network(network=args.network,
                  depth=args.depth,
                  dataset=args.dataset,
                  kaiming=args.kaiming)
origpar = parameter_count(net)
print('Original weight count:', origpar)
if args.rank_scale or args.target_ratio:
    if args.network == 'vgg':
        names = [str(i) for i, child in enumerate(net.feature) if i and type(child) == nn.Conv2d]
        denoms = [child.out_channels*child.kernel_size[0]*child.kernel_size[1] 
                  for child in net.feature if type(child) == nn.Conv2d]
        def compress(model, rank_scale, spectral=False, kaiming=False):
            for name, denom in zip(names, denoms):
                patch_module(model.feature, name, FactorizedConv,
                             rank_scale=rank_scale,
                             init='spectral' if spectral else 'kaiming' if kaiming else lambda X: nn.init.normal_(X, 0., math.sqrt(2. / denom)))
            return model
        no_decay = names if args.wd2fd else []
        skiplist = [] if args.wd2fd else names
    else:
        def compress(model, rank_scale, spectral=False, kaiming=False):
            blocks = [block for layer in list(model.children())[2:-1] for block in layer]
            names = [['conv1', 'conv2'] for _ in blocks]
            for block, namelist in zip(blocks, names):
                if hasattr(block, 'downsample') and not block.downsample is None:
                    namelist.append('downsample.0')
            for module, namelist in zip(blocks, names):
                for name in namelist:
                    patch_module(module, name, FactorizedConv,
                                 rank_scale=rank_scale,
                                 init='spectral' if spectral else 'kaiming' if kaiming else lambda X: nn.init.kaiming_normal_(X))
            return model
        no_decay = ['.conv1', '.conv2', 'downsample'] if args.wd2fd else []
        skiplist = [] if args.wd2fd else ['.conv1', '.conv2', 'downsample']
    if args.target_ratio:
        if args.spectral or args.kaiming:
            _, rank_scale = compress_model(net, compress, args.target_ratio)
            compress(net, rank_scale, spectral=args.spectral, kaiming=args.kaiming)
        else:
            net, _ = compress_model(net, compress, args.target_ratio)
    else:
        compress(net, args.rank_scale, spectral=args.spectral)
    newpar = parameter_count(net)
    print('Compressed weight count:', newpar)
    print('Compression ratio:', newpar / origpar)
else:
    no_decay, skiplist = [], []

torch.cuda.set_device(args.device)
net = net.to(args.device)

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler
optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() 
                    if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in net.named_parameters() 
                    if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
]
optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)
lr_schedule = {0: args.learning_rate,
               int(args.epoch*0.5): args.learning_rate*0.1,
               int(args.epoch*0.75): args.learning_rate*0.01}
lr_scheduler = PresetLRScheduler(lr_schedule)
# lr_scheduler = #StairCaseLRScheduler(0, args.decay_every, args.decay_ratio)

# init criterion
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0
savefile = os.path.join(args.log_dir, 'checkpoint.t7')
if args.resume or (args.auto_resume and os.path.isfile(savefile)):
    print('==> Resuming from checkpoint..')
    if args.resume:
        assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('checkpoint/pretrain/%s_%s%s_bn_best.t7' % (args.dataset, args.network, args.depth))
    else:
        checkpoint = torch.load(savefile)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter
#log_dir = os.path.join(args.log_dir, '%s_%s%s' % (args.dataset,
#                                                  args.network,
#                                                  args.depth))
log_dir = args.log_dir
makedirs(log_dir)
writer = SummaryWriter(log_dir)
with open(os.path.join(log_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_lr(optimizer), epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        frobdecay(net, coef=args.weight_decay, skiplist=skiplist)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    if args.rank_scale or args.target_ratio:
        frobnorm, nonorth = [], []
        for module in (module for module in net.modules() if hasattr(module, 'frobgrad')):
            U, VT = module.U.data, module.VT.data
            nonorth.append(sum(non_orthogonality(U, VT)) / 2.)
            frobnorm.append(frobenius_norm(U, VT))
        writer.add_scalar('metric/FrobNorm', sum(frobnorm) / len(frobnorm), epoch)
        writer.add_scalar('metric/NonOrth', sum(nonorth) / len(nonorth), epoch)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (lr_scheduler.get_lr(optimizer), test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'best_acc': acc,
        'epoch': epoch+1,
        'loss': loss,
        'args': args
    }
    if acc > best_acc:
        torch.save(state, os.path.join(args.log_dir, 'best.t7'))
#        if not os.path.isdir('checkpoint'):
#            os.mkdir('checkpoint')
#        if not os.path.isdir('checkpoint/pretrain'):
#            os.mkdir('checkpoint/pretrain')
#        torch.save(state, './checkpoint/pretrain/%s_%s%s_best.t7' % (args.dataset,
#                                                                     args.network,
#                                                                     args.depth))
        best_acc = acc
    state['best_acc'] = best_acc
    torch.save(state, os.path.join(args.log_dir, 'checkpoint.t7'))
    return acc, best_acc


for epoch in range(start_epoch, args.epoch):
    train(epoch)
    acc, best = test(epoch)


writer.flush()
with open(os.path.join(log_dir, 'results.json'), 'w') as f:
    json.dump({'final validation accuracy': acc,
               'best validation accuracy': best,
               'original parameter count': origpar,
               'compressed parameter count': newpar if args.rank_scale or args.target_ratio else origpar,
               'compression ratio': newpar / origpar if args.rank_scale or args.target_ratio else 1.},
               f, indent=4)
