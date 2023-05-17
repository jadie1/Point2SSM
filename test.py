import os
import sys
import importlib
import argparse
import logging
import munch
import yaml
import numpy as np
import torch
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

from dataset import MeshDataset, DPC_Dataset
from utils.train_utils import *

def test(test_set):
    if args.model_name == 'dpc':
        dataset_test = DPC_Dataset(args, test_set, scale_factor=args.scale_factor, ref_path=args.ref_path)
    else:
        dataset_test = MeshDataset(args, test_set, scale_factor=args.scale_factor)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    device = 'cuda:0'
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)
    net.load_state_dict(torch.load(args.best_model_path, map_location=device)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()
    # summary(net, [(1024,3), (1024,3)])

    metrics = ['cd_p', 'cd_t']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    idx_to_plot = [0,1]
    
    logging.info('Testing '+test_set+'...')

    if args.save_predictions:
        pred_dir = os.path.join(log_dir, test_set)
        save_output_path = os.path.join(pred_dir, 'output')
        save_input_path = os.path.join(pred_dir, 'input')
        save_gt_path = os.path.join(pred_dir, 'gt')
        os.makedirs(save_output_path, exist_ok=True)
        os.makedirs(save_input_path, exist_ok=True)
        os.makedirs(save_gt_path, exist_ok=True)
   
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if args.model_name[:3] == 'dpc':
                pc, ref, gt, names = data
                pc, ref, gt = pc.to(device), ref.to(device), gt.to(device)
                inputs, target = pc.contiguous(), ref.contiguous()
                result_dict = net(inputs, target, gt, is_training=False)
            else:
                pc, gt, names = data
                pc, gt = pc.to(device), gt.to(device)
                inputs = pc.contiguous() 
                result_dict = net(inputs, gt, is_training=False)

            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            if args.save_predictions:
                for j in range(len(names)):
                    np.savetxt(os.path.join(save_output_path, names[j]+'.particles'), result_dict['recon'][j].cpu().numpy()*args.scale_factor)
                    np.savetxt(os.path.join(save_input_path, names[j]+'.particles'), inputs[j].cpu().numpy()*args.scale_factor)
                    np.savetxt(os.path.join(save_gt_path, names[j]+'.particles'), gt[j].cpu().numpy()*args.scale_factor)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-t', '--test_set', help='train or test', default='all')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    if 'missing_percent' not in args:
        args['missing_percent'] = 0

    if not args.best_model_path:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.best_model_path)
    log_dir = os.path.dirname(args.best_model_path)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    if arg.test_set == 'all':
        for test_set in ['train', 'val', 'test']:
            test(test_set)
    else:
        test(arg.test_set)
