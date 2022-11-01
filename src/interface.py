import ctypes 
from numpy.ctypeslib import ndpointer
import itertools
import platform
import numpy as np
import json
import os

from constraints import check_buffer_usage, check_area_usage

failure_stats = dict()

class LevelConfig:
    def __init__(self, label, buf_size, num_sub_clusters, tile_sizes, spatial_dim):
        self.label = label
        self.buf_size = buf_size
        self.num_sub_clusters = num_sub_clusters
        self.tile_sizes = tile_sizes
        self.spatial_dim = spatial_dim

    def __repr__(self):
        return self.label + \
               "\nbuf_size: " + str(self.buf_size) + \
               "\nnum_subcluster: " + str(self.num_sub_clusters) + \
               "\ntile_sizes: " + str(self.tile_sizes) + \
               "\nspatial_dim: " + str(self.spatial_dim) + '\n'


def _verify_input_constraints(args, shape, num_simd_lanes, bit_width, bandwidth, level_configs):
    buffer_usage = check_buffer_usage(num_simd_lanes, bit_width, bandwidth, level_configs)
    area_usage = check_area_usage(args, num_simd_lanes, bit_width, bandwidth, level_configs)

    tile_valid = True
    for i in range(len(level_configs) - 1):
        for k in level_configs[i].tile_sizes.keys():
            tile_valid &= (level_configs[i].tile_sizes[k] >= level_configs[i+1].tile_sizes[k])

    valid_status = {}
    for k, v in buffer_usage.items():
        valid_status[k] = (v <= 2)
    for k, v in area_usage.items():
        valid_status[k] = (v <= 2)
    valid_status['tile_size'] = tile_valid

    return valid_status

def get_eval_func(args):
    if platform.system() == 'Linux':
        spotlight = ctypes.CDLL(os.path.join('build', 'libspotlight.so'))
    elif platform.system() == 'Darwin':
        spotlight = ctypes.CDLL(os.path.join('build', 'libspotlight.dylib'))
    else:
        assert(False)

    if args.dump_all:
        evaluate = spotlight.evaluateWithDump
    else:
        evaluate = spotlight.evaluate

    evaluate.argtypes= (
        ctypes.POINTER(ctypes.c_ulonglong),    # shape
        ctypes.c_char_p,       # layer_type
        ctypes.c_ulonglong,    # num_pes (computed below)
        ctypes.c_ulonglong,    # num_simd_lanes
        ctypes.c_ulonglong,    # bit_width
        ctypes.c_ulonglong,    # bandwidth
        ctypes.c_ulonglong,    # num_levels
        ctypes.POINTER(ctypes.c_ulonglong),    # buf_sizes
        ctypes.POINTER(ctypes.c_ulonglong),    # num_sub_clusters
        ctypes.c_char_p,                       # dataflow
        ctypes.c_ulonglong,    # search_permutations
        ctypes.c_char_p,    # logfile
    )

    if args.dump_all:
        evaluate.restype = ctypes.c_char_p
    else:
        evaluate.restype = ndpointer(dtype=ctypes.c_double, shape=(5,))

    return evaluate


def convert_args_and_invoke(args, eval_func, shape, num_simd_lanes, bit_width, bandwidth, dataflow, level_configs):
    global failure_stats

    assert(len(shape[1]) == 7 and len(shape[2]) == 7)
    if dataflow == 'searched':
        for level_config in level_configs:
            assert(len(level_config.tile_sizes) == 7)

    buf_sizes = [x.buf_size for x in level_configs]
    num_sub_clusters = [x.num_sub_clusters for x in level_configs]

    dataflow_list = list()
    tile_order_default = ['N', 'K', 'C', 'X', 'Y', 'R', 'S']
    if dataflow == 'searched':
        for i, level_config in enumerate(level_configs):
            s_dim = level_config.spatial_dim
            tile_size = level_config.tile_sizes[s_dim]
            dataflow_list.append('S' + s_dim + '|' + str(tile_size))
            for x in tile_order_default:
                if x == s_dim: continue
                tile_size = level_config.tile_sizes[x]
                dataflow_list.append(('T' + x + '|' + str(tile_size)))
            if i+1 < len(level_configs):
                level_configs[i+1].tile_sizes[s_dim] = min(level_configs[i+1].tile_sizes[s_dim], level_configs[i].tile_sizes[s_dim])
                dataflow_list.append('C')
    elif dataflow == 'eye':
        dataflow_list.append('TC|' + str(level_configs[0].tile_sizes['C']))
        dataflow_list.append('TK|' + str(level_configs[0].tile_sizes['K']))
        dataflow_list.append('SY\'|' + str(level_configs[1].num_sub_clusters))
        dataflow_list.append('TX\'|' + str(shape[1]['S']))
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        dataflow_list.append('C')
        dataflow_list.append('TC|1')
        dataflow_list.append('SY\'|1')
        dataflow_list.append('SX\'|1')
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        args.search_permutations = False
    elif dataflow == 'shi':
        dataflow_list.append('TK|' + str(level_configs[0].tile_sizes['K']))
        dataflow_list.append('SY\'|' + str(shape[1]['R']))
        dataflow_list.append('TX|' + str(level_configs[1].num_sub_clusters))
        dataflow_list.append('TC|' + str(level_configs[0].tile_sizes['C']))
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        dataflow_list.append('C')
        dataflow_list.append('TC|1')
        dataflow_list.append('TY\'|1')
        dataflow_list.append('SX\'|1')
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        args.search_permutations = False
    elif dataflow == 'dla':
        dataflow_list.append('SK|' + str(level_configs[0].tile_sizes['K']))
        dataflow_list.append('TC|' + str(level_configs[1].num_sub_clusters))
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        dataflow_list.append('TY|' + str(shape[1]['R']))
        dataflow_list.append('TX|' + str(shape[1]['S']))
        dataflow_list.append('C')
        dataflow_list.append('SC|1')
        dataflow_list.append('TY|' + str(shape[1]['R']))
        dataflow_list.append('TX|' + str(shape[1]['S']))
        dataflow_list.append('TR|' + str(shape[1]['R']))
        dataflow_list.append('TS|' + str(shape[1]['S']))
        args.search_permutations = False
    shape_list = list(itertools.chain(*[(shape[1][x], shape[2][x]) for x in tile_order_default]))
    layer_type = shape[3]
    # TODO: DSCONV causes seg fault (likely because dataflow requirements are different)
    layer_type = 'CONV'

    shape_array_type = ctypes.c_ulonglong * len(shape_list)
    level_array_type = ctypes.c_ulonglong * len(level_configs)
    dataflow_array_type = ctypes.c_ulonglong * len(dataflow_list)

    num_pes = np.product([l.num_sub_clusters for l in level_configs])

    logpath = os.path.join('logs', shape[0] + '.log')

    ret = eval_func(
        shape_array_type(*shape_list),
        ctypes.create_string_buffer(layer_type.encode('utf-8')),
        ctypes.c_ulonglong(num_pes),
        ctypes.c_ulonglong(num_simd_lanes),
        ctypes.c_ulonglong(bit_width),
        ctypes.c_ulonglong(bandwidth),
        ctypes.c_ulonglong(len(level_configs)),
        level_array_type(*buf_sizes),
        level_array_type(*num_sub_clusters),
        ctypes.create_string_buffer(','.join(dataflow_list).encode('utf-8')),
        ctypes.c_ulonglong(args.search_permutations),
        ctypes.create_string_buffer(logpath.encode('utf-8')),
    )

    if args.dump_all:
        cost = json.loads(ret.decode('utf-8'))
    else:
        cost = {
            'ExactRunTime': ret[0],
            'OverallEnergy': ret[1],
            'Area': ret[2],
            'Power': ret[3],
            'Throughput': ret[4]
        }

    if cost['ExactRunTime'] <= 0 or cost['OverallEnergy'] <= 0 or cost['Area'] <= 0:
        failure_stats['maestro'] = failure_stats.get('maestro', 0) + 1
        return None

    if cost['Area'] > args.max_area:
        failure_stats['area'] = failure_stats.get('area', 0) + 1
        return None

    if cost['Power'] > args.max_area:
        failure_stats['power'] = failure_stats.get('power', 0) + 1
        return None

    # if (args.target == 'delay' or args.target == 'throughput') and cost['OverallEnergy'] > constraint[0]:
    #     failure_stats['energy'] = failure_stats.get('energy', 0) + 1
    #     return None

    return cost
