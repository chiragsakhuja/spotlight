import numpy as np

# Input constraints:
#   ** Tile sizes cannot exceed buffer sizes. Expected buffer size is computed as
#     2 * (num_sub_clusters + 1) * mapped_volume_size
#   ** Product of all num_sub_clusters must be equivalent to total number of PEs
#   ** Tile sizes at a lower level (i.e. closer to PE) must be less than or equal
#        to tile size in the higher (i.e. closer to DRAM) cluster level
#   ** Area budget is met

def check_buffer_usage(num_simd_lanes, bit_width, bandwidth, level_configs):
    buffer_usage = {}

    for level, level_config in enumerate(level_configs):
        actual_sizes = dict()
        for dim in level_config.tile_sizes.keys():
            if dim in ['X', 'Y']:
                unroll_factor = level_config.num_sub_clusters if dim == level_config.spatial_dim else 1
                actual_sizes[dim] = level_config.tile_sizes[dim] + unroll_factor - 1
            else:
                unroll_factor = level_config.num_sub_clusters if dim == level_config.spatial_dim else 1
                actual_sizes[dim] = level_config.tile_sizes[dim] * unroll_factor

        requested_inp_size = 2 * (
            actual_sizes['N'] * actual_sizes['C'] *
            actual_sizes['X'] * actual_sizes['Y']
        )
        requested_wgt_size = 2 * (
            actual_sizes['K'] * actual_sizes['C'] *
            actual_sizes['R'] * actual_sizes['S']
        )
        requested_out_size = 2 * (
            actual_sizes['N'] * actual_sizes['K'] *
            (actual_sizes['X'] - min(actual_sizes['R'], actual_sizes['X']) + 1) *
            (actual_sizes['Y'] - min(actual_sizes['S'], actual_sizes['Y']) + 1)
        )

        buffer_usage['inp_valid_%d' % level] = (requested_inp_size / level_config.inp_buf_size)
        buffer_usage['wgt_valid_%d' % level] = (requested_wgt_size / level_config.wgt_buf_size)
        buffer_usage['out_valid_%d' % level] = (requested_out_size / level_config.out_buf_size)

    return buffer_usage


def check_area_usage(args, num_simd_lanes, bit_width, bandwidth, level_configs):
    area_usage = {}

    sram_areas = [0] * len(level_configs)
    bus_areas = [0] * len(level_configs)
    noc_areas = [0] * len(level_configs)
    area_per_l1_byte = 4505.1889 / 64
    area_per_l2_byte = 4161.536 / 32768
    for level, level_config in enumerate(level_configs):
        sram_multiplier = area_per_l1_byte if (level == len(level_configs) - 1) else area_per_l2_byte
        sram_areas[level] += sram_multiplier * (
                    level_config.inp_buf_size + level_config.wgt_buf_size + level_config.out_buf_size) * (bit_width / 8)
        bus_areas[level] += 14.662 * level_config.num_sub_clusters + 28.895
        noc_areas[level] += (1.2886 * (
                    level_config.num_sub_clusters ** 2) + 5.5814 * level_config.num_sub_clusters - 23.711) * bandwidth * 101.79

    num_pes = np.product([l.num_sub_clusters for l in level_configs])

    area_per_mac = 4470.9014
    compute_area = area_per_mac * num_simd_lanes * num_pes * ((bit_width / 8) ** 2)
    sram_area = sum(sram_areas)
    bus_area = sum(bus_areas)
    noc_area = sum(noc_areas)

    total_area = compute_area + sram_area + bus_area + noc_area

    # print("{} {:e} {:e} {:e} {:e}".format(num_pes, compute_area, sram_area, bus_area, noc_area))
    # print("{:e}".format(total_area))

    AREA_CONSTRAINT = args.max_area
    area_usage['area'] = total_area / AREA_CONSTRAINT
    return area_usage
