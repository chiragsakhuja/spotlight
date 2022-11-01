import numpy as np
import interface
from constraints import check_buffer_usage, check_area_usage
import math

class Parameter:
    def __init__(self, name, range):
        self.name = name
        self.range = range
    def __repr__(self):
        return '%s %s' % (self.name, str(self.range))

class Sample:
    def __init__(self, point, feats):
        self.point = point
        self.feats = feats

    def __repr__(self):
        return 'point {} feats {}'.format(
            self.point,
            self.feats
        )

class SWSample(Sample):
    def __init__(
        self,
        point,
        feats=None,
        cost=None,
        edp=None,
        energy=None,
        delay=None,
        throughput=None,
        area=None,
        power=None
    ):
        super().__init__(point, feats)
        if cost:
            self.edp = cost['OverallEnergy'] * cost['ExactRunTime']
            self.energy = cost['OverallEnergy']
            self.delay = cost['ExactRunTime']
            self.throughput = cost['Throughput']
            self.area = cost['Area']
            self.power = cost['Power']
        else:
            self.edp = edp
            self.energy = energy
            self.delay = delay
            self.throughput = throughput
            self.area = area
            self.power = power

    def __add__(self, o):
        return SWSample(None, None, None,
            self.edp + o.edp,
            self.energy + o.energy,
            self.delay + o.delay,
            self.throughput + o.throughput,
            self.area + o.area,
            self.power + o.power
        )

    def getResultString(self):
        return 'edp {:.4e} energy {:.4e} delay {:.4e} throughput {:.4e} area {:.4e} power {:4e}'.format(
            self.edp,
            self.energy,
            self.delay,
            self.throughput,
            self.area,
            self.power
        )

class HWSample(Sample):
    def __init__(self, point, feats, model_results, init_target_value, reduce_f, eval_f):
        super().__init__(point, feats)
        self.target_value = init_target_value
        self.reduce_f = reduce_f
        self.eval_f = eval_f
        self.samples = list()
        for layer_results in model_results:
            self.target_value = self.reduce_f(self.target_value, layer_results.opt_target_value)
            self.samples.append(layer_results.opt_sample)

    def getResultString(self):
        sum = SWSample(None, None, None, 0, 0, 0, 0, 0, 0)
        for x in self.samples:
            sum = sum + x
        return sum.getResultString()

    def __repr__(self):
        return 'target {:.4e} {} < {} >'.format(
            self.eval_f(self.target_value),
            super().__repr__(),
            ' | '.join(
                ['sw_results {} sw_point {} sw_feats {}'.format(
                    x.getResultString(),
                    str(x.point),
                    str(x.feats)) for x in self.samples
                ]
            )
        )

class Results:
    def __init__(self, init_opt_target_value, select_f, reduce_f, eval_f):
        self.select_f = select_f
        self.reduce_f = reduce_f
        self.eval_f = eval_f

        self.values = list()
        self.feats = list()

        self.opt_target_value = init_opt_target_value
        self.opt_sample = None

    def add(self, sample):
        cur_target_value = self.select_f(sample)
        status, new_opt_target_value = self.reduce_f(self.opt_target_value, cur_target_value)
        if status:
            self.opt_target_value = new_opt_target_value
            self.opt_sample = sample
        self.values.append(self.eval_f(cur_target_value))
        self.feats.append(sample.feats)

    def __repr__(self):
        return 'target {:.4e} {} point {} feats {}'.format(
            self.eval_f(self.opt_target_value),
            self.opt_sample.getResultString(),
            str(self.opt_sample.point),
            str(self.opt_sample.feats)
        )

class SWResults(Results):
    pass

class HWResults(Results):
    def __repr__(self):
        return '{} < {} >'.format(
            super().__repr__(),
            ' | '.join(
                ['sw_result {} sw_point {} sw_feats {}'.format(
                    x.getResultString(),
                    str(x.point),
                    str(x.feats)) for x in self.opt_sample.samples
                ]
            )
        )


def convert_point_to_maestro(args, hw_point, sw_point, num_levels):
    num_simd_lanes = hw_point.get('num_simd_lane')
    bit_width = hw_point.get('bit_width')
    bandwidth = hw_point.get('bandwidth')
    subclusters = hw_point.get('subclusters')

    buf_partition_counts = list()
    total = 1
    for i in range(num_levels):
        buf_partition_counts.append(total)
        total *= subclusters[num_levels - i - 1]
    buf_partition_counts.reverse()

    aggregate_tile_sizes = dict()
    if args.dataflow == 'searched':
        tiled_dims = ['N', 'K', 'C', 'X', 'Y', 'R', 'S']
    elif args.dataflow == 'fixed':
        tiled_dims = ['K', 'C']

    for dim in tiled_dims:
        aggregate_tile_sizes[dim] = np.multiply.accumulate(sw_point.get(dim))

    level_configs = list()
    for i in range(num_levels):
        buf_size_per_partition = int(hw_point.get('l{}_buf_size'.format(i)) / buf_partition_counts[i])
        if args.dataflow == 'searched':
            spatial_dim = sw_point.get('l{}_spatial_dim'.format(i))
        elif args.dataflow == 'fixed':
            spatial_dim = None
        level_configs.append(interface.LevelConfig(
            'L{}'.format(i),
            buf_size_per_partition,
            subclusters[i],
            {dim: aggregate_tile_size[i] for dim, aggregate_tile_size in aggregate_tile_sizes.items()},
            spatial_dim
        ))

    dataflow = 'searched' if args.dataflow == 'searched' else sw_point.get('dataflow')

    # TODO: Remove reverse
    level_configs.reverse()
    return num_simd_lanes, bit_width, bandwidth, dataflow, level_configs

def run_maestro_tvm(args, eval_func, shape, hw_point, sw_point, num_levels):
    num_simd_lanes, bit_width, bandwidth, dataflow, level_configs = convert_point_to_maestro(args, hw_point, sw_point, num_levels)
    return interface.convert_args_and_invoke(args, eval_func, shape, num_simd_lanes, bit_width, bandwidth, dataflow, level_configs)

def get_hw_point_feats(hw_point, num_levels, with_labels=False):
    feats = list()
    feats.append(hw_point.get('num_simd_lane'))
    feats.append(hw_point.get('bit_width'))
    feats.append(hw_point.get('bandwidth'))
    feats.append(np.sum([hw_point.get('l{}_buf_size'.format(i)) / 32768 for i in range(num_levels)]))
    feats.append(np.product(hw_point.get('subclusters')))
    feats.append(hw_point.get('subclusters')[0])

    if with_labels:
        feat_labels = ['num_simd_lane', 'bit_width', 'bandwidth', 'total_buf_size', 'total_pes', 'l0_pes']
        feat_labels = [f'hw_feat_{x}' for x in feat_labels]
        return feats, feat_labels
    return feats

def get_sw_point_feats(hw_point, sw_point, num_levels, excluded_feats, dataflow, with_labels=False):
    feats = list()

    if with_labels:
        feat_labels = list()

    if dataflow == 'searched':
        include_raw = not 'raw' in excluded_feats
        include_original = not 'original' in excluded_feats
        include_intuitive = not 'intuitive' in excluded_feats
        include_data_driven = not 'data-driven' in excluded_feats

        if include_original:
            subcluster_utilization = list()
            iterations = list()

        spatial_dim_shapes = dict()

        for i in range(num_levels):
            spatial_dim = sw_point.get('l{}_spatial_dim'.format(i))
            spatial_tiles = sw_point.get(spatial_dim)

            if not spatial_dim in spatial_dim_shapes:
                spatial_dim_shapes[spatial_dim] = np.product(spatial_tiles)

            if include_original:
                num_subclusters = hw_point.get('subclusters')[i]

                # Compute maximum number of subclusters that could be fully utilized
                degree_parallelism = math.floor(spatial_tiles[i + 1] / spatial_tiles[i])
                # Take ratio with actual number of subclusters to get real utilization
                actual_utilization = min(1.0, degree_parallelism / num_subclusters)
                subcluster_utilization.append(actual_utilization)

                spatial_width = spatial_tiles[i + 1] * spatial_tiles[i]
                iterations.append(math.ceil(spatial_width / num_subclusters))

        if include_original:
            feats.append(np.product(iterations))
            feats.append(np.product(subcluster_utilization))
            feats.append(sw_point.get('R')[-1] * sw_point.get('S')[-1])
            if with_labels:
                feat_labels.append('total_iterations')
                feat_labels.append('total_utilization')
                feat_labels.append('dram_kernel_tile')

        if include_original or include_intuitive:
            feats.append(np.product(list(spatial_dim_shapes.values())))
            if with_labels:
                feat_labels.append('shape')

        if include_data_driven:
            feats.append(
                2  * sw_point.get('X')[-1] +
                3  * sw_point.get('Y')[-1] +
                5  * sw_point.get('K')[-1] +
                7  * sw_point.get('K')[-2] +
                11 * sw_point.get('K')[-3]
            )
            feats.append(
                (sw_point.get('X')[-1] / sw_point.get('X')[0]) *
                (sw_point.get('Y')[-1] / sw_point.get('Y')[0]) *
                (hw_point.get('subclusters')[0] + hw_point.get('subclusters')[1])
            )
            if with_labels:
                feat_labels.append('data_driven_1')
                feat_labels.append('data_driven_2')

        if include_raw:
            for x in ['N', 'K', 'C', 'X', 'Y', 'R', 'S']:
                tiles = sw_point.get(x)
                for i in range(num_levels + 1):
                    feats.append(tiles[i])
                    if with_labels:
                        feat_labels.append(f'{x}{i}')
            for i in range(num_levels):
                feats.append(ord(sw_point.get('l{}_spatial_dim'.format(i))[0]))
                if with_labels:
                    feat_labels.append(f'l{i}_spatial_dim')
    elif dataflow == 'fixed':
        for x in ['K', 'C']:
            tiles = sw_point.get(x)
            for i in range(num_levels + 1):
                feats.append(tiles[i])
                if with_labels:
                    feat_labels.append(f'{x}{i}')

    if with_labels:
        feat_labels = [f'sw_feat_{x}' for x in feat_labels]
        return feats, feat_labels
    return feats