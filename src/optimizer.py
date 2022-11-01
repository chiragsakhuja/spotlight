from operator import itemgetter
import json
import os
import time

import tqdm
import numpy as np

import bo
import ga
import space
import search_utils

import sklearn.exceptions
import sklearn.inspection as insp
import scipy.stats as stats

class Optimizer:
    def __init__(self, args, eval_f, shapes, n_hw, n_sw, out_file, compute_feats=True):
        self.args = args
        self.eval_f = eval_f
        self.shapes = shapes
        self.n_hw = n_hw
        self.n_sw = n_sw
        self.out_file = out_file
        self.excluded_feats = args.exclude_feat.split(',')
        self.compute_feats = compute_feats
        self.hw_opt_complete_hook = None
        self.sw_opt_complete_hook = None
        self.sw_feat_labels = None
        self.hw_feat_labels = None

    def log(self, format, *args):
        if self.out_file:
            self.out_file.write(format.format(*args) + '\n')
        else:
            print(format.format(*args))

    def _edp_reduce(x, y):
        edp_x = x[0] * x[1]
        edp_y = y[0] * y[1]
        if edp_x < edp_y: return False, x
        elif edp_x > edp_y: return True, y
        else:
            if x[2] < y[2]: return False, x
            else: return True, y

    def _single_reduce(x, y):
        if x[0] < y[0]: return False, x
        elif x[0] > y[0]: return True, y
        else:
            if x[1] < y[1]: return False, x
            else: return True, y

    def evaluate_point(self, shape, hw_point, sw_point, num_levels):
        cost = search_utils.run_maestro_tvm(self.args, self.eval_f, shape, hw_point, sw_point, num_levels)
        return cost

    def opt_sw(self, num_levels, hw_point):
        model_results = list()
        model_status = True

        for i, shape in enumerate(self.shapes):

            if self.args.target == 'edp':
                layer_results = search_utils.SWResults(
                    (float('inf'), float('inf'), float('inf')),
                    lambda x: (x.energy, x.delay, x.area),
                    Optimizer._edp_reduce,
                    lambda x: x[0] * x[1]
                )
            elif self.args.target == 'delay':
                layer_results = search_utils.SWResults(
                    (float('inf'), float('inf')),
                    lambda x: (x.delay, x.area),
                    Optimizer._single_reduce,
                    lambda x: x[0]
                )

            sw_space = space.create_software_space(self.args, shape[1], num_levels)
            invalid_sample_count = 0
            valid_sample_count = 0
            self.reset_sw_state(sw_space, hw_point)

            if self.args.sw_progress_bar:
                pbar = tqdm.tqdm(total=self.n_sw)

            layer_start_time = time.perf_counter()

            while valid_sample_count < self.n_sw:
                sample_start_time = time.perf_counter()
                sw_point = self.get_sw_point(sw_space, hw_point, layer_results)
                if self.compute_feats:
                    sw_feats, self.sw_feat_labels = search_utils.get_sw_point_feats(hw_point, sw_point, num_levels, self.excluded_feats, self.args.dataflow, with_labels=True)
                else:
                    sw_feats = list()
                cost = self.evaluate_point(shape, hw_point, sw_point, num_levels)
                success = not cost is None
                self.increment_sw_opt(success)

                sample_end_time = time.perf_counter()
                if success:
                    sw_sample = search_utils.SWSample(sw_point, sw_feats, cost)
                    if self.args.print_sw_samples:
                        self.log('         {} sw_sample {} {} t {} sec', valid_sample_count, sw_sample.getResultString(), str(sw_sample), sample_end_time - sample_start_time)
                    if self.args.sw_progress_bar:
                        pbar.update(1)
                    layer_results.add(sw_sample)
                    valid_sample_count += 1
                else:
                    invalid_sample_count += 1
                if invalid_sample_count >= self.args.max_invalid:
                    model_status = False
                    break

            layer_end_time = time.perf_counter()

            if model_status:
                self.log('      {} opt_layer {} t {} sec', i, str(layer_results), layer_end_time - layer_start_time)
                model_results.append(layer_results)
            else:
                self.log('      {} opt_layer INVALID t {} sec', i, layer_end_time - layer_start_time)

            if self.args.sw_progress_bar:
                pbar.close()

            if self.sw_opt_complete_hook:
                self.sw_opt_complete_hook(self, shape, sw_space, hw_point, layer_results)

        return model_results if model_status else None

    def opt_hw(self):
        if self.args.target == 'edp':
            hw_results = search_utils.HWResults(
                (float('inf'), float('inf'), 0),
                lambda x: x.target_value,
                Optimizer._edp_reduce,
                lambda x: x[0] * x[1]
            )
        elif self.args.target == 'delay':
            hw_results = search_utils.HWResults(
                (float('inf'), 0),
                lambda x: x.target_value,
                Optimizer._single_reduce,
                lambda x: x[0]
            )

        hw_space = space.create_hardware_space(self.args)
        hw_status = True
        invalid_sample_count = 0
        valid_sample_count = 0

        self.reset_hw_state(hw_space)

        if self.args.hw_progress_bar:
            pbar = tqdm.tqdm(total=self.n_hw)

        total_start_time = time.perf_counter()

        while valid_sample_count < self.n_hw:
            sample_start_time = time.perf_counter()

            hw_point = self.get_hw_point(hw_space, hw_results)
            hw_feats, self.hw_feat_labels = search_utils.get_hw_point_feats(hw_point, hw_space.num_levels, with_labels=True)
            model_results = self.opt_sw(hw_space.num_levels, hw_point)
            success = not model_results is None
            self.increment_hw_opt(success)

            sample_end_time = time.perf_counter()

            if success:
                if self.args.target == 'edp':
                    hw_sample = search_utils.HWSample(
                        hw_point,
                        hw_feats,
                        model_results,
                        (0, 0, 0),
                        lambda x, y:  (x[0] + y[0], x[1] + y[1], max(x[2], y[2])),
                        lambda x: x[0] * x[1],
                    )
                elif self.args.target == 'delay':
                    hw_sample = search_utils.HWSample(
                        hw_point,
                        hw_feats,
                        model_results,
                        (0, 0),
                        lambda x, y:  (x[0] + y[0], max(x[1], y[1])),
                        lambda x: x[0],
                    )
                self.log('   {} hw_sample {} t {} sec', valid_sample_count, hw_sample, sample_end_time - sample_start_time)
                if self.args.hw_progress_bar:
                    pbar.update(1)
                hw_results.add(hw_sample)
                valid_sample_count += 1
            else:
                self.log('   {} hw_sample INVALID t {} sec', valid_sample_count, sample_end_time - sample_start_time)
                invalid_sample_count += 1

            if invalid_sample_count >= self.args.max_invalid:
                hw_status = False
                break

        total_end_time = time.perf_counter()

        if hw_status:
            self.log('opt_hw {} t {} sec', str(hw_results), total_end_time - total_start_time)
        else:
            self.log('opt_hw INVALID t {} sec', total_end_time - total_start_time)

        if self.args.hw_progress_bar:
            pbar.close()

        if self.hw_opt_complete_hook:
            self.hw_opt_complete_hook(self, hw_space, hw_results)

        return hw_results if hw_status else None

    def increment_hw_opt(self, success):
        pass

    def get_hw_point(self, hw_space, hw_results):
        return None

    def reset_hw_state(self, hw_space):
        pass

    def increment_sw_opt(self, success):
        pass

    def get_sw_point(self, sw_space, hw_point, layer_results):
        return None

    def reset_sw_state(self, sw_space, hw_point):
        pass

class RandomOptimizer(Optimizer):
    def get_hw_point(self, hw_space, hw_results):
        space_idx = np.random.randint(hw_space.size)
        return hw_space.build_point(space_idx)

    def get_sw_point(self, sw_space, hw_point, layer_results):
        space_idx = np.random.randint(sw_space.size)
        return sw_space.build_point(space_idx)

class GridOptimizer(Optimizer):
    def increment_hw_opt(self, status):
        self.hw_idx = (self.hw_idx + 1) % self.n_hw

    def get_hw_point(self, hw_space, hw_results):
        if not self.hw_idx:
            self.hw_idx = np.random.randint(hw_space.size)
        return hw_space.build_point(self.hw_idx)

    def reset_hw_state(self, hw_space):
        self.hw_idx = None

    def increment_sw_opt(self, success):
        self.sw_idx = (self.sw_idx + 1) % self.n_sw

    def get_sw_point(self, sw_space, hw_point, layer_results):
        if not self.sw_idx:
            self.sw_idx = np.random.randint(sw_space.size)
        return sw_space.build_point(self.sw_idx)

    def reset_sw_state(self, sw_space, hw_points):
        self.sw_idx = None

class GeneticOptimizer(Optimizer):
    def gen_hw_batch(self, hw_space, hw_results):
        if hw_results:
            self.hw_points_last_gen = [x for i, x in enumerate(self.hw_points) if self.hw_point_valid[i]]
            valid_count = sum([1 if x else 0 for x in self.hw_point_valid])
            self.hw_points_last_gen_f = hw_results.values[-valid_count:]
        self.hw_points = ga.generate_batch(
            hw_space,
            self.args.hw_batch_size,
            self.hw_points_last_gen,
            self.hw_points_last_gen_f
        )
        self.hw_idx = 0
        self.hw_point_valid = [True] * self.args.hw_batch_size

    def increment_hw_opt(self, success):
        self.hw_idx += 1

    def get_hw_point(self, hw_space, hw_results):
        if self.hw_idx >= self.args.hw_batch_size:
            self.gen_hw_batch(hw_space, hw_results)
        return self.hw_points[self.hw_idx]

    def reset_hw_state(self, hw_space):
        self.hw_points = list()
        self.hw_idx = 0
        self.hw_point_valid = [True] * self.args.hw_batch_size
        self.hw_points_last_gen = None
        self.hw_points_last_gen_f = None
        self.gen_hw_batch(hw_space, None)

    def gen_sw_batch(self, sw_space, hw_point, layer_results):
        if layer_results:
            self.sw_points_last_gen = [x for i, x in enumerate(self.sw_points) if self.sw_point_valid[i]]
            valid_count = sum([1 if x else 0 for x in self.sw_point_valid])
            self.sw_points_last_gen_f = layer_results.values[-valid_count:]
        self.sw_points = ga.generate_batch(
            sw_space,
            self.args.sw_batch_size,
            self.sw_points_last_gen,
            self.sw_points_last_gen_f
        )
        self.sw_idx = 0
        self.sw_point_valid = [True] * self.args.sw_batch_size

    def increment_sw_opt(self, success):
        self.sw_point_valid[self.sw_idx] = success
        self.sw_idx += 1

    def get_sw_point(self, sw_space, hw_point, layer_results):
        if self.sw_idx >= self.args.sw_batch_size:
            self.gen_sw_batch(sw_space, hw_point, layer_results)
        return self.sw_points[self.sw_idx]

    def reset_sw_state(self, sw_space, hw_point):
        self.sw_points = list()
        self.sw_idx = 0
        self.sw_point_valid = [True] * self.args.sw_batch_size
        self.sw_points_last_gen = None
        self.sw_points_last_gen_f = None
        self.gen_sw_batch(sw_space, hw_point, None)

class CoBOOptimizer(Optimizer):
    def __init__(self, args, eval_f, shapes, n_hw, n_sw, out_file, compute_feats=True):
        super().__init__(args, eval_f, shapes, n_hw, n_sw, out_file, compute_feats)
        self.sw_opt_complete_hook = self.sw_opt_complete
        self.hw_opt_complete_hook = self.hw_opt_complete

    def score(self, y_pred, y_truth):
        y_pred_order = np.argsort(y_pred.flatten())
        y_truth_order = np.argsort(y_truth.flatten())
        return stats.spearmanr(y_pred_order, y_truth_order)

    def analyze_bo(self, bo_inst, X, y, shape, labels):
        try:
            X_std = bo_inst.scale_x.transform(X)
            y_col = np.atleast_2d(y).T
            y_std = bo_inst.scale_y.transform(y_col)
        except sklearn.exceptions.NotFittedError:
            return

        s = self.score(bo_inst.model.predict(X_std), y_std)
        r = insp.permutation_importance(bo_inst.predictor, X_std, y_std, n_repeats=10, random_state=0)
        feats = list()
        for i in r.importances_mean.argsort()[::-1]:
            mean = r.importances_mean[i]
            std = r.importances_std[i]
            # if mean - 2 * std > 0:
            feats.append((i, mean, std))
        print('BO ', end='')
        if shape:
            print(shape[0], end='')
        print(' SpearmanR=', s, 'significant feats=', [(labels[x[0]], x[1], x[2]) for x in feats])

    def hw_opt_complete(self, _, hw_space, hw_results):
        if self.args.print_bo_analysis:
            self.analyze_bo(self.hw_bo, hw_results.feats, hw_results.values, None, self.hw_feat_labels)

    def sw_opt_complete(self, _, shape, sw_space, hw_point, layer_results):
        if self.args.print_bo_analysis:
            self.analyze_bo(self.sw_bo, layer_results.feats, layer_results.values, shape, self.sw_feat_labels)

    def gen_hw_batch(self, hw_space, hw_results):
        if hw_results:
            self.hw_bo.update(hw_results.feats, hw_results.values)
            self.hw_bo.fit()
        self.hw_points, feats = bo.generate_hw_batch(hw_space, self.args.hw_batch_size)
        self.hw_sort_index = self.hw_bo.run(feats)
        self.hw_idx = 0
        self.hw_valid_count = 0

    def increment_hw_opt(self, success):
        self.hw_idx += 1
        if success:
            self.hw_valid_count += 1

    def get_hw_point(self, hw_space, hw_results):
        if self.hw_valid_count >= self.args.hw_batch_trials:
            self.gen_hw_batch(hw_space, hw_results)
        return self.hw_points[self.hw_sort_index[self.hw_idx]]

    def reset_hw_state(self, hw_space):
        self.hw_bo = bo.BO(self.args)
        self.hw_points = list()
        self.hw_sort_idx = list()
        self.hw_idx = self.args.hw_batch_trials
        self.hw_valid_count = 0
        self.gen_hw_batch(hw_space, None)

    def gen_sw_batch(self, sw_space, hw_point, layer_results):
        if layer_results:
            self.sw_bo.update(layer_results.feats, layer_results.values)
            self.sw_bo.fit()
        self.sw_points, feats = bo.generate_sw_batch(sw_space, hw_point, self.args.sw_batch_size, self.excluded_feats, self.args.dataflow)
        self.sw_sort_idx = self.sw_bo.run(feats)
        self.sw_idx = 0
        self.sw_valid_count = 0

    def increment_sw_opt(self, success):
        self.sw_idx += 1
        if success:
            self.sw_valid_count += 1

    def get_sw_point(self, sw_space, hw_point, layer_results):
        if self.sw_valid_count >= self.args.sw_batch_trials:
            self.gen_sw_batch(sw_space, hw_point, layer_results)
        return self.sw_points[self.sw_sort_idx[self.sw_idx]]

    def reset_sw_state(self, sw_space, hw_point):
        self.sw_bo = bo.BO(self.args, warmup_iters=30, exploration_ratio=0.3)
        self.sw_points = list()
        self.sw_sort_idx = list()
        self.sw_idx = self.args.sw_batch_trials
        self.sw_valid_count = 0
        self.gen_sw_batch(sw_space, hw_point, None)

class HyperMapper(Optimizer):

    def __init__(self, args, eval_f, shapes, n_hw, n_sw, out_file, compute_feats=True):
        super().__init__(args, eval_f, shapes, n_hw, n_sw, out_file, compute_feats)
        self.init_hw_files = True
        self.init_sw_files = True
        self.optimizer_results = list()

    def serialize_space(self, space):
        space_dict = dict()
        space_dict['input_parameters'] = dict()
        input_params = space_dict['input_parameters']
        for param in space.params:
            if param.name == 'bit_width':
                continue
            input_params[param.name] = dict()
            param_entry = input_params[param.name]
            if type(param.range[0]) is int:
                if param.range[-1] - param.range[0] + 1 == len(param.range):
                    param_entry['parameter_type'] = 'integer'
                    param_entry['values'] = [param.range[0], param.range[-1]]
                else:
                    param_entry['parameter_type'] = 'ordinal'
                    param_entry['values'] = param.range
            else:
                param_entry['parameter_type'] = 'categorical'
                param_entry['values'] = [str(x) for x in param.range]
        return space_dict

    def sw_evaluator(self, X, hw_point, shape, num_levels):
        for k in X.keys():
            if type(X[k]) is str:
                if X[k][0] == '[' or X[k][0].isnumeric():
                    X[k] = eval(X[k])
        cost = self.evaluate_point(shape, hw_point, X, num_levels)
        if cost is None:
            fail_val = np.finfo(np.float64).max
            if self.args.target == 'edp':
                self.optimizer_results.append((fail_val, fail_val, fail_val))
            else:
                self.optimizer_results.append(fail_val)
            return fail_val
        else:
            delay = cost['ExactRunTime']
            energy = cost['OverallEnergy']
            if self.args.target == 'edp':
                self.optimizer_results.append((delay, energy, delay * energy))
                return delay * energy
            elif self.args.target == 'delay':
                self.optimizer_results.append(delay)
                return delay

    def opt_sw(self, num_levels, hw_point):
        from hypermapper import optimizer

        total_delay = 0
        total_energy = 0
        for shape in self.shapes:
            sw_file_path = os.path.join(self.args.output_dir, f'{shape[0]}.json')
            output_file_path = os.path.join(self.args.output_dir, f'{shape[0]}.csv')
            if self.init_sw_files:
                sw_space = space.create_software_space(self.args, shape[1], num_levels)
                config_dict = self.serialize_space(sw_space)
                config_dict['application_name'] = shape[0]
                config_dict['optimization_objectives'] = ['Value']
                config_dict['optimization_iterations'] = self.n_sw
                # config_dict['output_data_file'] = output_file_path
                with open(sw_file_path, 'w') as fd:
                    fd.write(json.dumps(config_dict))
            self.optimizer_results = list()
            optimizer.optimize(sw_file_path, lambda X: self.sw_evaluator(X, hw_point, shape, num_levels))
            if self.args.target == 'edp':
                min_sample = min(self.optimizer_results, key=itemgetter(2))
                total_delay += min_sample[0]
                total_energy += min_sample[1]
            elif self.args.target == 'delay':
                total_delay += min(self.optimizer_results)

        self.init_sw_files = False
        if self.args.target == 'edp':
            return total_delay * total_energy
        elif self.args.target == 'delay':
            return total_delay

    def hw_evaluator(self, num_levels, X):
        for k in X.keys():
            if type(X[k]) is str:
                if X[k][0] == '[' or X[k][0].isnumeric():
                    X[k] = eval(X[k])
        X['bit_width'] = 8
        return self.opt_sw(num_levels, X)

    def opt_hw(self):
        from hypermapper import optimizer

        hw_file_path = os.path.join(self.args.output_dir, 'HW.json')
        output_file_path = os.path.join(self.args.output_dir, self.args.output_filename)
        hw_space = space.create_hardware_space(self.args)
        if self.init_hw_files:
            config_dict = self.serialize_space(hw_space)
            config_dict['application_name'] = 'HW'
            config_dict['optimization_objectives'] = ['Value']
            config_dict['optimization_iterations'] = self.n_hw
            config_dict['output_data_file'] = output_file_path
            with open(hw_file_path, 'w') as fd:
                fd.write(json.dumps(config_dict))
            self.init_hw_files = False
        optimizer.optimize(hw_file_path, lambda X: self.hw_evaluator(hw_space.num_levels, X))

class Exhaustive(Optimizer):
    def opt_sw(self, num_levels, hw_point):
        for i, shape in enumerate(self.shapes):
            sw_space = space.create_software_space(self.args, shape[1], num_levels)
            param_names = [x.name for x in sw_space.params]
            dimension_names = np.repeat(param_names[0:7], num_levels + 1)
            dimension_labels = [x[0] + str(x[1]) for x in zip(dimension_names, list(range(num_levels + 1)) * 7)]
            self.log('shape {} {}', shape[0], str(shape[1]))
            self.log('sw_idx,{},{},energy,delay,throughput,area',
                ','.join(dimension_labels),
                ','.join(param_names[7:])
            )
            for i in range(int(self.n_sw)):
                idx = np.random.randint(sw_space.size)
                sw_point = sw_space.build_point(idx)
                cost = search_utils.run_maestro_tvm(self. args, self.eval_f, shape, hw_point, sw_point, num_levels)
                if cost:
                    self.log('{},{},{},{},{},{},{}',
                        idx,
                        ','.join([','.join([str(y) for y in x]) for x in sw_point.param_values[0:7]]),
                        ','.join([str(x) for x in sw_point.param_values[7:]]),
                        cost['OverallEnergy'],
                        cost['ExactRunTime'],
                        cost['Throughput'],
                        cost['Area']
                    )

    def opt_hw(self):
        hw_space = space.create_hardware_space(self.args)
        assert(self.args.exhaustive_hw_end_idx > self.args.exhaustive_hw_start_idx)
        print(hw_space.size)

        for i in range(self.n_hw):
            idx = self.args.exhaustive_hw_start_idx + np.random.randint(self.args.exhaustive_hw_end_idx - self.args.exhaustive_hw_start_idx)
            idx %= hw_space.size
            hw_point = hw_space.build_point(idx)
            self.log('hw_sample idx {} {}', idx, str(hw_point))
            self.opt_sw(hw_space.num_levels, hw_point)
