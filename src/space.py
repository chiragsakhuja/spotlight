import numpy as np

class Parameter:
    def __init__(self, name, range):
        self.name = name
        self.range = range
    def __repr__(self):
        return '%s %s' % (self.name, str(self.range))

class Point:
    def __init__(self, params=dict()):
        self.param_labels = list()
        self.param_values = list()
        for label, value in params.items():
            self.add(label, value)

    def add(self, label, value):
        self.param_labels.append(label)
        self.param_values.append(value)

    def set(self, label, value):
        try:
            idx = self.param_labels.index(label)
            self.param_values[idx] = value
        except ValueError:
            self.add(label, value)

    def get(self, label):
        try:
            idx = self.param_labels.index(label)
            return self.param_values[idx]
        except ValueError:
            return None

    def __repr__(self):
        ret = str(dict(zip(self.param_labels, self.param_values)))
        return ret.replace(' ', '')

class Space:
    def __init__(self, params, num_levels):
        self.params = params
        self.num_levels = num_levels
        self.build_meta()

    def build_meta(self):
        self.param_length = [len(x.range) for x in self.params]
        self.cumulative_length = np.flip(np.multiply.accumulate(np.flip(self.param_length)))
        self.size = self.cumulative_length[0]
        self.cumulative_length = self.cumulative_length[1:]

    def build_point(self, index):
        point = Point()
        for i in range(len(self.params) - 1):
            point.add(self.params[i].name, self.params[i].range[int(index / self.cumulative_length[i])])
            index %= self.cumulative_length[i]
        point.add(self.params[-1].name, self.params[-1].range[index % len(self.params[-1].range)])
        return point

def get_all_combinations(n, V, ret, curr):
    if n == 0:
        return
    if n == 1:
        ret.append(curr + [int(V)])
    possible_values = [v for v in range(int(V), 0, -1) if V % v == 0]
    for pv in possible_values:
        get_all_combinations(n-1, V/pv, ret, curr+[pv])
    return ret

def get_all_combinations_v2(n, V, ret, curr):
    if n == 0:
        return
    if n == 1:
        ret.append(curr + [int(V)])
    possible_values = [v for v in range(2, int(V)+1) if V % v == 0]
    for pv in possible_values:
        get_all_combinations_v2(n-1, V/pv, ret, curr+[pv])
    return ret

def get_all_summation(n, V, step, ret, curr):
    if n == 0:
        return
    if n == 1:
        ret.append(curr + [int(V)])
    possible_values = [v for v in range(step, V, step) if v < V]
    for pv in possible_values:
        get_all_summation(n-1, V-pv, step, ret, curr+[pv])
    return ret

def create_hardware_space(args, num_element_type=3):
    # TODO: now randomly choose number of levels, not sure if this is the right thing to do
    # num_levels = np.random.randint(args.levels_low, args.levels_high+1)
    num_levels = 2

    params = list()
    # print('Creating hardware resources...')
    params.append(Parameter('num_simd_lane', list(range(args.simd_low, args.simd_high+1, args.simd_step))))
    params.append(Parameter('bit_width', list(range(args.prec_low, args.prec_high+1, args.prec_step))))
    params.append(Parameter('bandwidth', list(range(args.bw_low, args.bw_high+1, args.bw_step))))

    arg_dict = vars(args)
    for num_level in range(num_levels):
        step = arg_dict['l{}_step'.format(num_level+1)]
        low = arg_dict['l{}_low'.format(num_level+1)]
        high = arg_dict['l{}_high'.format(num_level+1)]
        buf_range = [1024 * v for v in range(low, high+1, step)]
        params.append(Parameter('l{}_buf_size'.format(num_level), buf_range))

    # subcluster
    step = args.pe_step
    pe_range = range(args.pe_low, args.pe_high+1, step)
    subcluster_range = []
    for num_pe in pe_range:
        subcluster_range += list(get_all_combinations_v2(num_levels, num_pe, [], []))
    params.append(Parameter('subclusters', subcluster_range))

    return Space(params, num_levels)

def create_software_space(args, shape, num_levels):
    # print('Creating dataflows...')
    params = []
    # tile sizes
    params.append(Parameter('K', list(get_all_combinations(num_levels+1, shape['K'], [], []))))
    params.append(Parameter('C', list(get_all_combinations(num_levels+1, shape['C'], [], []))))
    if args.dataflow == 'searched':
        params.append(Parameter('N', list(get_all_combinations(num_levels+1, shape['N'], [], []))))
        params.append(Parameter('X', list(get_all_combinations(num_levels+1, shape['X'], [], []))))
        params.append(Parameter('Y', list(get_all_combinations(num_levels+1, shape['Y'], [], []))))
        params.append(Parameter('R', list(get_all_combinations(num_levels+1, shape['R'], [], []))))
        params.append(Parameter('S', list(get_all_combinations(num_levels+1, shape['S'], [], []))))

        # spatial_dim
        for num_level in range(num_levels):
            params.append(Parameter('l{}_spatial_dim'.format(num_level), ['K', 'C', 'X', 'Y', 'R', 'S']))
    elif args.dataflow == 'fixed':
        params.append(Parameter('dataflow', ['eye', 'dla', 'shi']))

    return Space(params, num_levels)