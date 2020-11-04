from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import models.networks as networks
from .base_model import BaseModel
import math
import os


class HyperRIMModel(BaseModel):
    def __init__(self, opt):
        super(HyperRIMModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.netG.train()
        self.load()
        # store the number of levels and code channel
        self.num_levels = int(math.log(opt['scale'], 2))
        self.code_nc = opt['network_G']['code_nc']
        self.map_nc = opt['network_G']['map_nc']

        # define losses, optimizer and scheduler
        self.netF = networks.define_F(opt).to(self.device)
        self.projections = None
        if self.is_train:
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            map_network_params = []
            core_network_params = []
            # can freeze weights for any of the levels
            freeze_level = train_opt['freeze_level']
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    if freeze_level:
                        if "level_%d" % freeze_level not in k:
                            if 'map' in k:
                                map_network_params.append(v)
                            else:
                                core_network_params.append(v)
                    else:
                        if 'map' in k:
                            map_network_params.append(v)
                        else:
                            core_network_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam([{'params': core_network_params},
                                                 {'params': map_network_params, 'lr': 1e-2 * train_opt['lr_G']}],
                                                lr=train_opt['lr_G'], weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # for resume training - load the previous optimizer stats
            self.load_optimizer()

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, train_opt['lr_steps'],
                                                                    train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, code=[], need_HR=True):
        self.lr = data['LR'].to(self.device)
        self.code = code
        if need_HR:  # train or val
            self.targets = dict()
            # only feed the images, not their paths
            for key, val in data.items():
                if ('HR' in key or 'D' in key) and 'path' not in key:
                    self.targets[key] = val.to(self.device)

    # Generate random code input at specified level (if left empty, then generate code for all levels)
    def gen_code(self, bs, w, h, levels=[], tensor_type=torch.randn):
        gen_levels = levels if levels != [] else range(self.num_levels)
        out_code = []
        for i in gen_levels:
            out_code.append(tensor_type(bs, self.map_nc + self.code_nc * w * (2 ** i) * h * (2 ** i)).to(self.device))
        return out_code

    # Random projection matrix for reducing LPIPS feature dimension
    def init_projection(self, h, total_dim=1000):
        # default to h == w
        fake_input = torch.zeros(1, 3, h, h)
        fake_feat, fake_shape = self.netF(fake_input)
        self.projections = []
        dim_per_layer = int(total_dim * 1. / len(fake_feat))
        for feat in fake_feat:
            self.projections.append(F.normalize(torch.randn(feat.shape[1], dim_per_layer), p=2, dim=1).to(self.device))

    def clear_projection(self):
        self.projections = None

    def _get_target_at_level(self, level):
        for key, val in self.targets.items():
            if str(level + 1) in key and 'path' not in key:
                return val
        return self.targets['HR']

    def compute_feature_loss(self, gen_feat, real_feat, gen_shape):
        # compute l2 feature loss given features
        result = 0
        for i, g_feat in enumerate(gen_feat):
            cur_diff = torch.sum((g_feat - real_feat[i]) ** 2, dim=1) / (gen_shape[i] ** 2)
            result += cur_diff
        return result

    def get_features(self, level=-1):
        '''
        Assuming the generated features are for the same LR input, therefore just one pass for the target feature
        '''
        self.netG.eval()
        out_dict = OrderedDict()
        with torch.no_grad():
            gen_imgs = self.netG(self.lr, self.code)
            gen_feat, gen_shape = self.netF(gen_imgs[level])
            real_feat, real_shape = self.netF(self._get_target_at_level(level))
            gen_features = []
            real_features = []
            # random projection
            for i, g_feat in enumerate(gen_feat):
                proj_gen_feat = torch.mm(g_feat, self.projections[i])
                proj_real_feat = torch.mm(real_feat[i], self.projections[i])
                gen_features.append(proj_gen_feat / gen_shape[i])
                real_features.append(proj_real_feat / gen_shape[i])
            gen_features = torch.cat(gen_features, dim=1)
            real_features = torch.cat(real_features, dim=1)

            out_dict['gen_feat'] = gen_features
            out_dict['real_feat'] = real_features

        self.netG.train()
        return out_dict

    def get_loss(self, level=-1):
        self.netG.eval()
        with torch.no_grad():
            gen_imgs = self.netG(self.lr, self.code)
            gen_feat, gen_shape = self.netF(gen_imgs[level])
            real_feat, real_shape = self.netF(self._get_target_at_level(level))
            result = self.compute_feature_loss(gen_feat, real_feat, gen_shape)
        self.netG.train()
        return result

    def optimize_parameters(self, step, inter_supervision=False):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer_G.zero_grad()
        # intermediate supervision adds loss from intermediate resolutions
        if inter_supervision:
            outputs = self.netG(self.lr, self.code)
            l_g_total = 0.
            for i, output in enumerate(outputs):
                gen_feat, gen_shape = self.netF(output)
                real_feat, real_shape = self.netF(self._get_target_at_level(i))
                l_g_total += self.compute_feature_loss(gen_feat, real_feat, gen_shape)
        else:
            outputs = self.netG(self.lr, self.code)
            l_g_total = 0.
            gen_feat, gen_shape = self.netF(outputs[-1])
            real_feat, real_shape = self.netF(self._get_target_at_level(-1))
            l_g_total += self.compute_feature_loss(gen_feat, real_feat, gen_shape)
        l_g_total = torch.sum(l_g_total)
        l_g_total.backward()
        self.optimizer_G.step()
        self.log_dict['l_g_lpips'] = l_g_total.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            output = self.netG(self.lr, self.code)
            self.pred = output[-1]
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.lr.detach()[0].float().cpu()
        out_dict['HR_pred'] = self.pred.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.targets['HR'].detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def load_optimizer(self):
        load_path_O = self.opt['path']['pretrain_model_O']
        if load_path_O is not None:
            print('loading optimizer [{:s}] ...'.format(load_path_O))
            self.optimizer_G.load_state_dict(torch.load(load_path_O))

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.optimizer_G, 'O', iter_label)
