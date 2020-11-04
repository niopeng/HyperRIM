import torch
import sys
import math
from dciknn_cuda.dciknn_cuda import DCI


def print_without_newline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def get_data_at_level(data, level):
    for key, val in data.items():
        if str(level) in key and 'path' not in key:
            return key, val
    return 'HR', data['HR']


def generate_code_samples(model, data, opt):
    options = opt['train']

    # For DCI
    dci_num_comp_indices = int(options['dci_num_comp_indices'])
    dci_num_simp_indices = int(options['dci_num_simp_indices'])
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])
    # Block and thread size for parallel CUDA programming
    block_size = 100 if 'block_size' not in options else options['block_size']
    thread_size = 10 if 'thread_size' not in options else options['thread_size']
    # Used for DCI query
    num_outer_iterations = 5000 if 'num_outer_iterations' not in options else options['num_outer_iterations']
    num_samples_per_img = int(options['num_samples_per_img'])
    num_levels = int(math.log(opt['network_G']['scale'], 2))

    sampled_codes = []
    sampled_targets = set()

    print("Generating Samples")
    with torch.no_grad():
        for level_num in range(1, num_levels + 1):
            num_instances = data['LR'].shape[0]
            torch.cuda.empty_cache()
            target_name, target_data = get_data_at_level(data, level_num)
            sampled_targets.add(target_name)
            project_dim = 1000 if 'project_dims' not in options else options['project_dims'][level_num - 1]
            mini_batch_size = 20 if 'mini_batch_size' not in options else options['mini_batch_size']
            # handle really large target resolution explicitly due to vRAM constraint
            if target_data.shape[-1] > 256:
                project_dim = 700
                mini_batch_size = 5
            model.init_projection(target_data.shape[-1], project_dim)
            dci_db = DCI(project_dim, dci_num_comp_indices, dci_num_simp_indices, block_size, thread_size)

            cur_sampled_code = model.gen_code(data['LR'].shape[0], data['LR'].shape[2], data['LR'].shape[3],
                                              levels=[level_num - 1], tensor_type=torch.empty)[0]

            for sample_index in range(num_instances):
                if (sample_index + 1) % 10 == 0:
                    print_without_newline('\rFinding level %d code: Processed %d out of %d instances' % (
                        level_num, sample_index + 1, num_instances))
                code_pool = model.gen_code(num_samples_per_img, data['LR'].shape[2], data['LR'].shape[3],
                                           levels=[level_num - 1])[0]
                feature_pool = []

                for i in range(0, num_samples_per_img, mini_batch_size):
                    cur_data = {key: data[key][sample_index] for key in sampled_targets}
                    cur_data['LR'] = data['LR'][sample_index].expand(mini_batch_size, -1, -1, -1)
                    # fix the previously sampled code
                    if len(code_pool.shape) > 2:
                        code_samples = [cur_code[sample_index].expand(mini_batch_size, -1, -1, -1)
                                        for cur_code in sampled_codes]
                    else:
                        code_samples = [cur_code[sample_index].expand(mini_batch_size, -1)
                                        for cur_code in sampled_codes]
                    # add the new samples
                    code_samples.append(code_pool[i:i + mini_batch_size])
                    model.feed_data(cur_data, code=code_samples)
                    feature_output = model.get_features(level=(level_num - 1))
                    feature_pool.append(feature_output['gen_feat'])

                feature_pool = torch.cat(feature_pool, dim=0)
                dci_db.add(feature_pool.reshape(num_samples_per_img, -1))
                target_feature = feature_output['real_feat']
                best_sample_idx, _ = dci_db.query(
                    target_feature.reshape(target_feature.shape[0], -1), 1, num_outer_iterations)
                cur_sampled_code[sample_index, :] = code_pool[int(best_sample_idx[0][0]), :]
                # clear the db
                dci_db.clear()

            print_without_newline('\rFinding level %d code: Processed %d out of %d instances\n' % (
                level_num, num_instances, num_instances))

            sampled_codes.append(cur_sampled_code)
            dci_db.free()

        torch.cuda.empty_cache()

        # add sample perturbations
        for i, sample in enumerate(sampled_codes):
            sampled_codes[i] = sample + model.gen_code(data['LR'].shape[0], data['LR'].shape[2], data['LR'].shape[3],
                                                       levels=[i])[0] * sample_perturbation_magnitude

    return sampled_codes
