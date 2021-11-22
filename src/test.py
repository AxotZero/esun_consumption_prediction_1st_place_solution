import argparse
import os
from pdb import set_trace as bp

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
# my lib
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from utils import to_device
from constant import target_indices


def main(config, output_type='top3_indices', output_dir='./submission.csv'):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['validation_split'] = False
    config['data_loader']['args']['training'] = False

    data_loader = getattr(module_data, config['data_loader']['type'])(
        **config['data_loader']['args'],
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    outputs_logits = []
    outputs_top3 = []
    chids = []
    with torch.no_grad():
        for i, (data, chid) in enumerate(tqdm(data_loader)):
            output = model(to_device(data, device))
            output = output[:, target_indices]
            _, output_topk_indices = torch.topk(output, 3, dim=1)

            chids.append(chid.cpu().detach().numpy())
            outputs_logits.append(output.cpu().detach().numpy())
            outputs_top3.append(target_indices[output_topk_indices.cpu().detach().numpy().astype(int)]+1)

    chids = np.concatenate(chids, axis=0).reshape(-1, 1)
    outputs_logits = np.concatenate(outputs_logits, axis=0)
    outputs_top3 = np.concatenate(outputs_top3, axis=0)
    # bp()
    pd.DataFrame(
        data=np.concatenate([chids, outputs_top3], axis=1),
        columns=['chid', 'top1', 'top2', 'top3']).to_csv(f'{output_dir}/outputs_top3.csv', index=None)
    pd.DataFrame(
        data=np.concatenate([chids, outputs_logits], axis=1),
        columns=['chid']+list(range(16))).to_csv(f'{output_dir}/outputs_logits.csv', index=None)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--output_type', type=str, default='top3_indices', choices=['top3_indices', 'logits'])
    args.add_argument('-o', '--output_dir', default='./submission.csv', type=str,
                      help='output_dir')

    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()
    output_type = args.output_type
    output_dir = args.output_dir
    main(config, output_type, output_dir)
