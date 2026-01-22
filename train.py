"""
Training script for CS-Net
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import visdom
import numpy as np
import argparse
from model.csnet import CSNet
from dataloader.rnfb import Data
from utils.train_metrics import metrics
from utils.visualize import init_visdom_line, update_lines
from utils.dice_loss_single_class import dice_coeff_loss

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Net Training Script")
    parser.add_argument("data_path", type=str, nargs='?', default="dataset/DRIVE/", help="Path to DRIVE dataset root directory (default: dataset/DRIVE/)")
    parser.add_argument("--rgb", action="store_true", help="Use RGB images (3 channels). If not set, use grayscale (1 channel).")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--snapshot", type=int, default=100, help="Snapshot interval for saving checkpoints (default: 100)")
    parser.add_argument("--test-step", type=int, default=1, help="Test step interval (default: 1)")
    parser.add_argument("--ckpt-path", type=str, default="checkpoint/", help="Directory to save checkpoints (default: 'checkpoint/')")
    parser.add_argument("--device-ids", type=str, default="0", help="Comma-separated list of GPU device IDs to use (default: '0')")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    return parser.parse_args()


def setup_visdom():
    X, Y = 0, 0.5  # for visdom
    x_acc, y_acc = 0, 0
    x_sen, y_sen = 0, 0
    env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
    env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
    env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
    return env, panel, env1, panel1, env2, panel2

def get_ckpt_prefix(data_path):
    # Remove trailing slash, replace / with _, remove file extension if any
    base = os.path.splitext(data_path.rstrip('/'))[0].replace('/', '_')
    return f'CS_Net_{base}'

def save_ckpt(net, iter, ckpt_path, ckpt_prefix):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_name = f'{ckpt_prefix}_{iter}.pkl'
    torch.save(net, os.path.join(ckpt_path, ckpt_name))
    print(f'--->saved model: {os.path.join(ckpt_path, ckpt_name)}<--- ')


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    # set the channels to 3 when the format is RGB, otherwise 1.

    channels = 3 if args.rgb else 1
    net = CSNet(classes=1, channels=channels).cuda()
    # Parse device_ids string to list of ints
    device_ids = [int(i) for i in args.device_ids.split(",") if i.strip() != ""]
    net = nn.DataParallel(net, device_ids=device_ids).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
    critrion = nn.MSELoss().cuda()
    # critrion = nn.CrossEntropyLoss().cuda()
    print("---------------start training------------------")
    # load train dataset
    train_data = Data(args.data_path, train=True, rgb=args.rgb)
    batchs_data = DataLoader(train_data, batch_size=args.batch_size, num_workers=2, shuffle=True)

    # env, panel, env1, panel1, env2, panel2 = setup_visdom()

    iters = 1
    accuracy = 0.
    sensitivty = 0.
    ckpt_prefix = get_ckpt_prefix(args.data_path)
    for epoch in range(args.epochs):
        net.train()
        for idx, batch in enumerate(batchs_data):
            image = batch[0].cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()
            pred = net(image)
            # pred = pred.squeeze_(1)
            loss1 = critrion(pred, label)
            loss2 = dice_coeff_loss(pred, label)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            acc, sen = metrics(pred, label, pred.shape[0])
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}'.format(epoch + 1,
                                                                                     iters, loss.item(),
                                                                                     acc / pred.shape[0],
                                                                                     sen / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            # X, x_acc, x_sen = iters, iters, iters
            # Y, y_acc, y_sen = loss.item(), acc / pred.shape[0], sen / pred.shape[0]
            # update_lines(env, panel, X, Y)
            # update_lines(env1, panel1, x_acc, y_acc)
            # update_lines(env2, panel2, x_sen, y_sen)
            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args.lr, iter=epoch, max_iter=args.epochs, power=0.9)
        if (epoch + 1) % args.snapshot == 0:
            save_ckpt(net, epoch + 1, args.ckpt_path, ckpt_prefix)

        # model eval
        if (epoch + 1) % args.test_step == 0:
            test_acc, test_sen = model_eval(net, args)
            print("Average acc:{0:.4f}, average sen:{1:.4f}".format(test_acc, test_sen))

            if (accuracy > test_acc) & (sensitivty > test_sen):
                save_ckpt(net, epoch + 1 + 8888888, args.ckpt_path, ckpt_prefix)
                accuracy = test_acc
                sensitivty = test_sen



def model_eval(net, args):
    print("Start testing model...")
    test_data = Data(args.data_path, train=False, rgb=args.rgb)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc, Sen = [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc, sen = metrics(pred_val, label, pred_val.shape[0])
        print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}".format(acc, sen))
        Acc.append(acc)
        Sen.append(sen)
        file_num += 1
        # for better view, add testing visdom here.
        return np.mean(Acc), np.mean(Sen)



if __name__ == '__main__':
    args = parse_args()
    train(args)
