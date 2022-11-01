#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "spotlight-common.hpp"

constexpr uint64_t fact(uint64_t n) {
  uint64_t ret = 1;
  for(uint64_t i = 2; i <= n; ++i) {
    ret *= i;
  }
  return ret;
}

void getShape(maestro::Options const & option, ShapeT & shape);
void getDataflow(
  maestro::Options const & option,
  ShapeT const & shape,
  Point & point,
  uint64_t & cluster_pos,
  uint64_t & tile_sweep_pos,
  char & tile_sweep_dim
);

int main(int argc, char** argv) 
{
  maestro::Options option;
  bool success = option.parse(argc, argv);

  if (!success) {
    std::cout << "[MAESTRO] Failed to parse program options" << std::endl;
  }

  std::string layer = option.analysis_layer;
  std::string dataflow = option.analysis_accelerator;
  std::string mode = option.analysis_mode;

  maestro::InitializeBaseObjects(option.message_print_lv);

  uint64_t cluster_pos = 0;
  uint64_t tile_sweep_pos = 0;
  char tile_sweep_dim = 0;

  Point point;
  ShapeT shape;
  std::string layer_type = "CONV";
  point.num_pes = option.np;
  point.num_simd_lanes = option.num_simd_lanes;
  point.l1_size = option.l1_size;
  point.l2_size = option.l2_size;
  point.bit_width = option.bit_width;
  point.bw = option.bw;
  point.latency = option.hops;

  getShape(option, shape);
  getDataflow(option, shape, point, cluster_pos, tile_sweep_pos, tile_sweep_dim);
  uint64_t space_id = 0;
  uint64_t space_size = 0;

  std::ofstream result_file;
#ifndef _DEBUG_OUT
  if(option.result_file != "") {
    result_file.open(option.result_file);
    result_file << "Delay,Energy,Area,Fitness,Dataflow\n";
  }
#endif

  std::vector<Point> space_batch;
  std::vector<Cost<false>> cost_batch;
  Cost<false> best_cost;
  Point best_point;

  std::cout << "Building space...";

  auto start = std::chrono::high_resolution_clock::now();
  if(mode == "tiles") {
    throw std::runtime_error("invalid mode selection");
#if 0
    for(uint64_t i = 4; i <= shape[tile_sweep_dim]; ++i) {
      Point copy = point;
      std::get<1>(copy.dataflow[tile_sweep_pos]) = std::max(i, 1ul);

      space_batch[batch_idx] = copy;
      ++batch_idx;
      if(batch_idx == option.n_parallel) {
        runBatch<false>(option.n_parallel, shape, layer_type, space_batch, cost_batch, result_file, best_point, best_cost, "");
        batch_idx = 0;
      }
    }
#endif
  } else if(mode == "permutations") {
    std::vector<uint64_t> high_indices(cluster_pos), low_indices(point.dataflow.size() - cluster_pos - 1);
    std::iota(high_indices.begin(), high_indices.end(), 0);
    std::iota(low_indices.begin(), low_indices.end(), 0);

    uint64_t batch_size = fact(high_indices.size()) * fact(low_indices.size());
    space_batch.reserve(batch_size);
    cost_batch.reserve(batch_size);

    do {
      do {
        Point copy = point;
        for(uint8_t i = 0; i < high_indices.size(); ++i) { copy.dataflow[i] = point.dataflow[high_indices[i]]; }
        for(uint8_t i = 0; i < low_indices.size(); ++i) { copy.dataflow[cluster_pos+1+i] = point.dataflow[cluster_pos+1+low_indices[i]]; }

        if(space_id >= option.space_id_end) { goto space_done; }
        if(space_id++ < option.space_id_start) { continue; }

        space_batch.push_back(copy);
        cost_batch.push_back(Cost<false>{});
        ++space_size;
      } while(std::next_permutation(std::begin(low_indices), std::end(low_indices)));
    } while(std::next_permutation(std::begin(high_indices), std::end(high_indices)));
  } else if(mode == "permutations_random") {
    static constexpr uint64_t num_samples = 1000;

    std::vector<uint64_t> indices(cluster_pos);
    std::iota(indices.begin(), indices.end(), 0);

    space_batch.resize(num_samples);
    cost_batch.resize(num_samples);

    std::random_device rd;
    std::mt19937 rng{rd()};
    for(uint64_t i = 0; i < num_samples; ++i) {
      Point copy = point;
      std::shuffle(indices.begin(), indices.end(), rng);

      for(uint64_t j = 0; j < cluster_pos; ++j) { copy.dataflow[j] = point.dataflow[indices[j]]; }

      space_batch[i] = copy;
      ++space_size;
    }
  } else if(mode == "single") {
    best_cost = run<false>(shape, layer_type, point, true, false, true, "log.txt");
    best_point = point;
    ++space_size;
  }
space_done:

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cout << duration.count() << "s (size: " << space_size << ")\nEvaluating space...";

  start = std::chrono::high_resolution_clock::now();
  runBatch(space_batch.size(), shape, layer_type, space_batch, cost_batch, result_file, best_point, best_cost, "");
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cout << duration.count() << "s\n";

  // if(! result_file.is_open()) {
  //   for(uint64_t i = 0; i < space_size; ++i) {
  //     std::cout << space_batch[i];
  //     printCost(std::cout, cost_batch[i]) << '\n';
  //   }
  // }

  std::cout << '\n' << best_point;
  printCost(std::cout, best_cost) << '\n';
  return 0;
}

void getShape(maestro::Options const & option, ShapeT & shape)
{
  (void) shape;
  std::string layer = option.analysis_layer;
  throw std::runtime_error("invalid layer selection");
}

void getDataflow(
  maestro::Options const & option,
  ShapeT const & shape,
  Point & point,
  uint64_t & cluster_pos,
  uint64_t & tile_sweep_pos,
  char & tile_sweep_dim
)
{
  std::string dataflow = option.analysis_accelerator;
  if(dataflow == "eyeriss_validation") {
    point.dataflow.push_back(std::make_tuple('S', 1, "Y'"));
    point.dataflow.push_back(std::make_tuple('T', 1, "X'"));
    point.dataflow.push_back(std::make_tuple('T', 1, "N"));
    point.dataflow.push_back(std::make_tuple('T', 2, "C"));
    point.dataflow.push_back(std::make_tuple('T', 16, "K"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('R').first, "R"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "S"));

    point.dataflow.push_back(std::make_tuple('C', shape.at('R').first, "P"));

    point.dataflow.push_back(std::make_tuple('S', 1, "Y"));
    point.dataflow.push_back(std::make_tuple('S', 1, "R"));
    point.dataflow.push_back(std::make_tuple('T', 1, "C"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "S"));

    cluster_pos = 7;
    tile_sweep_pos = 4;
    tile_sweep_dim = 'K';
  } else if(dataflow == "eyeriss") {
    point.dataflow.push_back(std::make_tuple('T', 2, "K"));
    point.dataflow.push_back(std::make_tuple('T', 4, "C"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('R').first, "R"));
    point.dataflow.push_back(std::make_tuple('S', shape.at('R').first, "Y"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "X"));

    point.dataflow.push_back(std::make_tuple('C', shape.at('R').first, "P"));

    point.dataflow.push_back(std::make_tuple('S', 1, "X"));
    point.dataflow.push_back(std::make_tuple('S', 1, "S"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "S"));

    cluster_pos = 5;
  } else if(dataflow == "nvdla") {
    point.dataflow.push_back(std::make_tuple('S', 1, "K"));
    point.dataflow.push_back(std::make_tuple('T', 64, "C"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('R').first, "R"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "S"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('R').first, "Y"));
    point.dataflow.push_back(std::make_tuple('T', shape.at('S').first, "X"));

    point.dataflow.push_back(std::make_tuple('C', 64, "P"));

    point.dataflow.push_back(std::make_tuple('S', 1, "C"));
    point.dataflow.push_back(std::make_tuple('T', 1, "R"));
    point.dataflow.push_back(std::make_tuple('T', 1, "S"));
    // CS TODO: remove the following line
    point.dataflow.push_back(std::make_tuple('T', 4, "K"));
    point.dataflow.push_back(std::make_tuple('T', 1, "Y"));
    point.dataflow.push_back(std::make_tuple('T', 1, "X"));

    // point.dataflow.push_back(std::make_tuple('T', shape['R'], "Y"));
    // point.dataflow.push_back(std::make_tuple('T', shape['S'], "X"));
    // point.dataflow.push_back(std::make_tuple('T', shape['R'], "R"));
    // point.dataflow.push_back(std::make_tuple('T', shape['S'], "S"));

    cluster_pos = 6;
    tile_sweep_pos = 0;
    tile_sweep_dim = 'K';
  } else if(dataflow == "verification") {
    point.dataflow.push_back(std::make_tuple('S', 1, "C"));
    point.dataflow.push_back(std::make_tuple('T', 2, "N"));
    point.dataflow.push_back(std::make_tuple('T', 8, "K"));
    point.dataflow.push_back(std::make_tuple('T', 31, "X"));
    point.dataflow.push_back(std::make_tuple('T', 1, "Y"));
    point.dataflow.push_back(std::make_tuple('T', 1, "R"));
    point.dataflow.push_back(std::make_tuple('T', 1, "S"));

    point.dataflow.push_back(std::make_tuple('C', 16, "P"));

    point.dataflow.push_back(std::make_tuple('S', 1, "K"));
    point.dataflow.push_back(std::make_tuple('T', 2, "N"));
    point.dataflow.push_back(std::make_tuple('T', 1, "C"));
    point.dataflow.push_back(std::make_tuple('T', 1, "X"));
    point.dataflow.push_back(std::make_tuple('T', 1, "Y"));
    point.dataflow.push_back(std::make_tuple('T', 1, "R"));
    point.dataflow.push_back(std::make_tuple('T', 1, "S"));

    cluster_pos = 7;
  } else {
    throw std::runtime_error("invalid dataflow selection");
  }
}