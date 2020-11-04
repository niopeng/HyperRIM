import os.path
import os
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger
from sampler import generate_code_samples
import numpy as np


def validate(val_loader, opt, model, current_step, epoch, logger):
    print('---------- validation -------------')
    start_time = time.time()

    avg_psnr = 0.0
    avg_lpips = 0.0
    idx = 0
    for val_data in val_loader:
        idx += 1
        img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
        img_dir = os.path.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        tensor_type = torch.zeros if opt['train']['zero_code'] else torch.randn
        code = model.gen_code(val_data['LR'].shape[0], val_data['LR'].shape[2], val_data['LR'].shape[3],
                              tensor_type=tensor_type)
        model.feed_data(val_data, code=code)
        model.test()

        visuals = model.get_current_visuals()
        sr_img = util.tensor2img(visuals['HR_pred'])  # uint8
        gt_img = util.tensor2img(visuals['HR'])  # uint8

        # Save generated images for reference
        save_img_path = os.path.join(img_dir, '{:s}_{:s}_{:d}.png'.format(opt['name'], img_name, current_step))
        util.save_img(sr_img, save_img_path)

        # calculate PSNR
        sr_img = sr_img
        gt_img = gt_img
        avg_psnr += util.psnr(sr_img, gt_img)

        avg_lpips += torch.sum(model.get_loss(level=-1))

    if current_step == 0:
        print('Saving the model at the end of iter {:d}.'.format(current_step))
        model.save(current_step)

    avg_psnr = avg_psnr / idx
    avg_lpips = avg_lpips / idx
    time_elapsed = time.time() - start_time
    # Save to log
    print_rlt = OrderedDict()
    print_rlt['model'] = opt['model']
    print_rlt['epoch'] = epoch
    print_rlt['iters'] = current_step
    print_rlt['time'] = time_elapsed
    print_rlt['psnr'] = avg_psnr
    print_rlt['lpips'] = avg_lpips
    logger.print_format_results('val', print_rlt)
    print('-----------------------------------')


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and
                 not key == 'pretrain_model_G'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size_per_month']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            num_months = int(opt['train']['num_months'])
            num_days = int(opt['train']['num_days'])
            total_iters = int(num_months * num_days)
            print('Total epochs needed: {:d} for iters {:,d}'.format(num_months, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
            batch_size_per_month = dataset_opt['batch_size_per_month']
            batch_size_per_day = int(opt['datasets']['train']['batch_size_per_day'])
            use_dci = opt['train']['use_dci']
            inter_supervision = opt['train']['inter_supervision']
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    validate(val_loader, opt, model, current_step, 0, logger)
    for epoch in range(num_months):
        for i, train_data in enumerate(train_loader):
            # Sample the codes used for training of the month
            if use_dci:
                cur_month_code = generate_code_samples(model, train_data, opt)
            else:
                tensor_type = torch.zeros if opt['train']['zero_code'] else torch.randn
                cur_month_code = model.gen_code(train_data['LR'].shape[0], train_data['LR'].shape[2],
                                                train_data['LR'].shape[3], tensor_type=tensor_type)
            # clear projection matrix to save memory
            model.clear_projection()
            for j in range(num_days):
                current_step += 1
                # get the sliced data
                cur_day_batch_start_idx = (j * batch_size_per_day) % batch_size_per_month
                cur_day_batch_end_idx = cur_day_batch_start_idx + batch_size_per_day
                if cur_day_batch_end_idx > batch_size_per_month:
                    cur_day_batch_idx = np.hstack((np.arange(cur_day_batch_start_idx, batch_size_per_month),
                                                   np.arange(cur_day_batch_end_idx - batch_size_per_month)))
                else:
                    cur_day_batch_idx = slice(cur_day_batch_start_idx, cur_day_batch_end_idx)

                cur_day_train_data = {key: val[cur_day_batch_idx] for key, val in train_data.items()}
                code = [gen_code[cur_day_batch_idx] for gen_code in cur_month_code]
                # training
                model.feed_data(cur_day_train_data, code=code)
                model.optimize_parameters(current_step, inter_supervision=inter_supervision)

                time_elapsed = time.time() - start_time
                start_time = time.time()

                # log
                if current_step % opt['logger']['print_freq'] == 0 or current_step == 1:
                    logs = model.get_current_log()
                    print_rlt = OrderedDict()
                    print_rlt['model'] = opt['model']
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = current_step
                    print_rlt['time'] = time_elapsed
                    for k, v in logs.items():
                        print_rlt[k] = v
                    print_rlt['lr'] = model.get_current_learning_rate()
                    logger.print_format_results('train', print_rlt)

                # save models
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    print('Saving the model at the end of iter {:d}.'.format(current_step))
                    model.save(current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    validate(val_loader, opt, model, current_step, epoch, logger)

                # update learning rate
                model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    main()
