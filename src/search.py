import space
import search_utils

import interface
import ga
import layers
import optimizer
import time

def log(out_file, format, *args):
    if out_file:
        out_file.write(format.format(*args) + '\n')
    else:
        print(format.format(*args))

def invoke_sw_optimizer(args, eval_func, shapes, num_levels, hw_point, out_file):
    sw_trials = args.sw_trials
    if 'bo' in args.model:
        opt = optimizer.CoBOOptimizer(args, eval_func, shapes, 0, sw_trials, out_file)
    elif 'ga' in args.model:
        opt = optimizer.GeneticOptimizer(args, eval_func, shapes, 0, sw_trials, out_file)
    elif 'grid' in args.model:
        opt = optimizer.GridOptimizer(args, eval_func, shapes, 0, sw_trials, out_file)
    elif 'random' in args.model:
        opt = optimizer.RandomOptimizer(args, eval_func, shapes, 0, sw_trials, out_file)
    elif 'hypermapper' in args.model:
        opt = optimizer.HyperMapper(args, eval_func, shapes, 0, sw_trials, out_file)
    else:
        assert(False)

    return opt.opt_sw(num_levels, hw_point)

def invoke_hw_optimizer(args, eval_func, shapes, out_file):
    hw_trials = args.hw_trials
    sw_trials = args.sw_trials
    if 'bo' in args.model:
        opt = optimizer.CoBOOptimizer(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    elif 'ga' in args.model:
        opt = optimizer.GeneticOptimizer(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    elif 'grid' in args.model:
        opt = optimizer.GridOptimizer(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    elif 'random' in args.model:
        opt = optimizer.RandomOptimizer(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    elif 'exhaustive' in args.model:
        opt = optimizer.Exhaustive(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    elif 'hypermapper' in args.model:
        opt = optimizer.HyperMapper(args, eval_func, shapes, hw_trials, sw_trials, out_file)
    else:
        assert(False)

    return opt.opt_hw()

def run_search(args, out_file):
    eval_func = interface.get_eval_func(args)

    shapes = layers.get_shapes(args.layers, args.ignore_stride, True, args.remove_duplicate_layers)
    assert(len(shapes) > 0)

    if args.hw_point:
        hw_templates = {
            'MoRV_delay': "{'num_simd_lane':16,'bit_width':8,'bandwidth':231,'l0_buf_size':122880,'l1_buf_size':98304,'subclusters':[9,32]}",
            'MoRV_edp': "{'num_simd_lane':16,'bit_width':8,'bandwidth':244,'l0_buf_size':237568,'l1_buf_size':122880,'subclusters':[33,9]}",
            'MoRVMnT_edp': "{'num_simd_lane':16,'bit_width':8,'bandwidth':185,'l0_buf_size':57344,'l1_buf_size':90112,'subclusters':[17,8]}",
            'MoRVMnT_delay': "{'num_simd_lane':16,'bit_width':8,'bandwidth':128,'l0_buf_size':40960,'l1_buf_size':147456,'subclusters':[5,59]}",
            'bo_edp_mnasnet': "{'num_simd_lane':16,'bit_width':8,'bandwidth':198,'l0_buf_size':122880,'l1_buf_size':32768,'subclusters':[14,21]}",
            'bo_edp_resnet': "{'num_simd_lane':16,'bit_width':8,'bandwidth':206,'l0_buf_size':81920,'l1_buf_size':73728,'subclusters':[9,33]}",
            'bo_edp_vgg': "{'num_simd_lane':16,'bit_width':8,'bandwidth':200,'l0_buf_size':65536,'l1_buf_size':73728,'subclusters':[33,9]}",
            'bo_edp_mobilenet': "{'num_simd_lane':15,'bit_width':8,'bandwidth':239,'l0_buf_size':139264,'l1_buf_size':32768,'subclusters':[56,5]}",
            'bo_edp_ztransformer': "{'num_simd_lane':16,'bit_width':8,'bandwidth':99,'l0_buf_size':188416,'l1_buf_size':73728,'subclusters':[299,1]}",
            'bo_delay_mnasnet': "{'num_simd_lane':16,'bit_width':8,'bandwidth':251,'l0_buf_size':122880,'l1_buf_size':147456,'subclusters':[15,20]}",
            'bo_delay_resnet': "{'num_simd_lane':16,'bit_width':8,'bandwidth':243,'l0_buf_size':237568,'l1_buf_size':221184,'subclusters':[17,16]}",
            'bo_delay_vgg': "{'num_simd_lane':16,'bit_width':8,'bandwidth':245,'l0_buf_size':81920,'l1_buf_size':106496,'subclusters':[33,9]}",
            'bo_delay_mobilenet': "{'num_simd_lane':16,'bit_width':8,'bandwidth':242,'l0_buf_size':106496,'l1_buf_size':81920,'subclusters':[34,7]}",
            'bo_delay_ztransformer': "{'num_simd_lane':16,'bit_width':8,'bandwidth':185,'l0_buf_size':163840,'l1_buf_size':212992,'subclusters':[3,83]}",
            'eyeriss_edge': "{'num_simd_lane':1,'bit_width':16,'bandwidth':256,'l0_buf_size':16384,'l1_buf_size':262144,'subclusters':[15,18]}",
            'eyeriss_cloud': "{'num_simd_lane':1,'bit_width':16,'bandwidth':256,'l0_buf_size':16777216,'l1_buf_size':268435456,'subclusters':[60,72]}",
            'nvdla_edge': "{'num_simd_lane':1,'bit_width':8,'bandwidth':256,'l0_buf_size':32768,'l1_buf_size':524288,'subclusters':[16,16]}",
            'nvdla_cloud': "{'num_simd_lane':1,'bit_width':8,'bandwidth':256,'l0_buf_size':1048576,'l1_buf_size':16777216,'subclusters':[120,120]}",
        }
        if args.hw_point in hw_templates:
            hw_point_str = hw_templates[args.hw_point]
        else:
            hw_point_str = args.hw_point

    if args.hw_point and args.sw_point:
        assert(len(shapes) == 1)
        runner = optimizer.Optimizer(args, eval_func, shapes, 1, 1, out_file)
        hw_point = space.Point(eval(hw_point_str))
        num_levels = len(hw_point.get('subclusters'))
        sw_point = space.Point(eval(args.sw_point))
        cost = runner.evaluate_point(shapes[0], hw_point, sw_point, num_levels)
        print(cost)
    elif args.hw_point:
        total_start_time = time.perf_counter()

        hw_point = space.Point(eval(hw_point_str))
        num_levels = len(hw_point.get('subclusters'))
        hw_feats = search_utils.get_hw_point_feats(hw_point, num_levels)

        model_results = invoke_sw_optimizer(args, eval_func, shapes, num_levels, hw_point, out_file)

        if model_results:
            if args.target == 'edp':
                hw_sample = search_utils.HWSample(
                    hw_point,
                    hw_feats,
                    model_results,
                    (0, 0, 0),
                    lambda x, y:  (x[0] + y[0], x[1] + y[1], max(x[2], y[2])),
                    lambda x: x[0] * x[1],
                )
                hw_results = search_utils.HWResults(
                    (float('inf'), float('inf'), 0),
                    lambda x: x.target_value,
                    optimizer.Optimizer._edp_reduce,
                    lambda x: x[0] * x[1]
                )
            elif args.target == 'delay':
                hw_sample = search_utils.HWSample(
                    hw_point,
                    hw_feats,
                    model_results,
                    (0, 0),
                    lambda x, y:  (x[0] + y[0], max(x[1], y[1])),
                    lambda x: x[0],
                )
                hw_results = search_utils.HWResults(
                    (float('inf'), 0),
                    lambda x: x.target_value,
                    optimizer.Optimizer._single_reduce,
                    lambda x: x[0]
                )

            hw_results.add(hw_sample)

            total_end_time = time.perf_counter()
            log(out_file, 'opt_hw {} t {} sec', str(hw_results), total_end_time - total_start_time)
    else:
        invoke_hw_optimizer(args, eval_func, shapes, out_file)
