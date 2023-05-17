import os
import sys
import yaml
import argparse
import logging
import math
import importlib
import datetime
import random
import munch
import time
import torch
import torch.optim as optim
import warnings
import shutil
import subprocess

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

from dataset import MeshDataset, DPC_Dataset
from utils.train_utils import *

def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    if args.model_name == 'dpc':
        dataset = DPC_Dataset(args, 'train')
        scale_factor = dataset.get_scale_factor()
        dataset_test = DPC_Dataset(args, 'val', scale_factor=scale_factor, ref_path=args.ref_path)
    else:
        dataset = MeshDataset(args, 'train')
        scale_factor = dataset.get_scale_factor()
        dataset_test = MeshDataset(args, 'val', scale_factor=scale_factor)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    if hasattr(model_module, 'weights_init'):
        net.apply(model_module.weights_init)

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    epochs_since_best_cd_t = 0
    for epoch in range(args.start_epoch, args.nepoch):
        start_time = time.time()
        torch.cuda.empty_cache()
        train_loss_meter.reset()
        net.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                out, loss = net(source, target, gt)
            else:
                pc, gt, names = data
                pc, gt = pc.to(device), gt.to(device)
                inputs = pc.contiguous()
                if args.model_name == 'cpae':
                    out, loss = net(inputs, gt, epoch=epoch)
                else:
                    out, loss = net(inputs, gt)


            train_loss_meter.update(loss.mean().item())
            loss.backward()
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss.mean().item(), lr) + ' time: ' + str(time.time()-start_time)[:4] + ' track: ' + str(epochs_since_best_cd_t) )

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)
            if args.early_stop:
                if best_cd_t:
                    epochs_since_best_cd_t = 0
                else:
                    if epoch > args.early_stop_start:
                        epochs_since_best_cd_t += 1
                if epochs_since_best_cd_t > args.early_stop_patience:
                    print("Early stopping epoch:", epoch)
                    break

    best_cd_t = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, device)

    args['best_model_path'] = log_dir+'/best_cd_p_network.pth'
    args['scale_factor'] = scale_factor
    return


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, device):
    best_cd_t = False
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                source, target = pc.contiguous(), ref.contiguous()
                result_dict = net(source, target, gt, is_training=False)
            else:
                pc, gt, names = data
                pc, gt = pc.to(device), gt.to(device)
                inputs = pc.contiguous() 
                result_dict = net(inputs, gt, is_training=False)

            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
                if loss_type == 'cd_t': # or loss_type =='kld': #TODO
                    best_cd_t = True
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)
    return best_cd_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    print_time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        if 'encoder' in args:
            exp_name = args.model_name+'_'+args.encoder
        else:
            exp_name = args.model_name
        exp_name += '_'+print_time.replace(':',"-")
        log_dir = os.path.join(args.work_dir, args.dataset, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    

    train()

    # Update yaml in log dir
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)
    print(os.path.join(log_dir, os.path.basename(config_path)))

    # Test
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path))])



