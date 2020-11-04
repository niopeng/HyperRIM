import os
import sys
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger
import torch

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['lpips'] = []
    test_results['psnr'] = []
    test_results['ssim'] = []

    for data in test_loader:
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
        multiple = 1 if "multiple" not in opt else opt["multiple"]

        # For generating multiple samples of the same input image
        for run_index in range(multiple):
            code = model.gen_code(data['LR'].shape[0], data['LR'].shape[2], data['LR'].shape[3])
            model.feed_data(data, code=code, need_HR=need_HR)
            model.test()

            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            visuals = model.get_current_visuals(need_HR=need_HR)
            sr_img = util.tensor2img(visuals['HR_pred'])  # uint8

            if need_HR:  # load target image and calculate metric scores
                gt_img = util.tensor2img(visuals['HR'])
                psnr = util.psnr(sr_img, gt_img)
                ssim = util.ssim(sr_img, gt_img, multichannel=True)
                lpips = torch.sum(model.get_loss(level=-1))
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['lpips'].append(lpips)
                print('{:20s} - LPIPS: {:.4f}; PSNR: {:.4f} dB; SSIM: {:.4f}.'.format(img_name, lpips, psnr, ssim))
            else:
                print(img_name)

            save_img_path = os.path.join(dataset_dir, img_name + '_%d.png' % run_index)
            util.save_img(sr_img, save_img_path)

    if need_HR:  # metrics
        # Average LPIPS/PSNR/SSIM results
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('----Average PSNR/SSIM results for {}----\n\tLPIPS: {:.4f}; PSNR: {:.4f} dB; SSIM: {:.4f}\n'
              .format(test_set_name, ave_lpips, ave_psnr, ave_ssim))
