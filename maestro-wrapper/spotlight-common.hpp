#ifndef _SPOTLIGHT_COMMON_HPP
#define _SPOTLIGHT_COMMON_HPP

#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ctpl_stl.h"

#include "AHW_noc-model.hpp"
#include "API_configuration.hpp"
#include "API_user-interface-v2.hpp"
#include "BASE_base-objects.hpp"
#include "BASE_constants.hpp"
#include "CA_cost-analysis-results.hpp"
#include "DFA_tensor.hpp"
#include "DSE_cost-database.hpp"
#include "DSE_csv_writer.hpp"
#include "DSE_design_point.hpp"
#include "DSE_hardware_modules.hpp"
#include "option.hpp"

#ifndef _DEBUG_OUT
  #define MULTICORE
#endif

using ShapeT = std::unordered_map<char, std::pair<uint64_t, uint64_t>>;
using DataflowT = std::vector<std::tuple<char, uint64_t, std::string>>;

struct Point
{
  uint64_t num_pes;
  uint64_t num_simd_lanes;
  uint64_t l1_size;
  uint64_t l2_size;
  uint64_t bit_width;
  uint64_t bw;
  uint64_t offchip_bw = 70000;
  uint64_t latency;
  DataflowT dataflow;
};

struct CostSmall
{
  double delay;
  double energy;
  double area;
  double power;
  double throughput;
  bool valid = false;

  double getFitness(void) const { return delay * energy; }
};

struct CostFull
{
  std::map<maestro::MetricType, long double> costs;
  bool valid = false;

  double getFitness(void) const
  {
    if(valid) { return costs.at(maestro::ExactRunTime) * costs.at(maestro::OverallEnergy); }
    else { return 0; }
  }
};

template<bool DumpAll>
struct Cost : std::conditional<DumpAll, CostFull, CostSmall>::type
{
  bool operator<(Cost const & other) const
  {
    if(this->valid && ! other.valid) { return true; }
    if(! this->valid && other.valid) { return false; }

    return this->getFitness() < other.getFitness();
  }

  bool operator==(Cost const & other) const
  {
    return this->valid == other.valid && this->getFitness() == other.getFitness();
  }
};

std::ostream & operator<<(std::ostream & out, Point const & point);

template<bool DumpFull>
std::ostream & printCost(std::ostream & out, Cost<DumpFull> const & cost)
{
  if constexpr (DumpFull) {
    (void) cost;
    out << "Not Implemented\n";
  } else {
#ifdef _DEBUG_OUT
  if(cost.valid) {
    out << "Delay: " << cost.delay << " cycles\n";
    out << "Energy: " << cost.energy << " x MAC energy\n";
    out << "Area: " << cost.area << " um^2\n";
    out << "Power: " << cost.power << " mW\n";
    out << "Throughput: " << cost.throughput << " MAC/cycle\n";
    out << "Fitness: " << cost.getFitness() << '\n';
  } else {
    out << "Invalid\n";
  }
#else
  if(cost.valid) {
    out << cost.delay << ',' << cost.energy << ',' << cost.area << ',' << cost.throughput << ',' << cost.getFitness();
  } else {
    out << "-1,-1,-1,-1";
  }
#endif
  }

  return out;
}


std::string MetricTypeToString(maestro::MetricType const & t);

std::shared_ptr<maestro::APIV2> configure(
  uint64_t num_pe,
  uint64_t num_simd_lanes,
  uint64_t l1_size,
  uint64_t l2_size,
  uint64_t bit_width,
  uint64_t bw,
  uint64_t offchip_bw,
  uint64_t hops,
  ShapeT const & shape,
  std::string const & layer_type,
  DataflowT const & dataflow
);

template<bool DumpAll>
Cost<DumpAll> run(ShapeT const & shape,
  std::string const & layer_type,
  Point const & point,
  bool print_results_to_screen,
  bool print_results_to_file,
  bool print_log_to_file,
  std::string logfile
)
{
  try {
    bool valid = true;
    auto api = configure(point.num_pes, point.num_simd_lanes, point.l1_size, point.l2_size, point.bit_width, point.bw, point.offchip_bw, point.latency, shape, layer_type, point.dataflow);
    auto res = api->AnalyzeNeuralNetwork(valid, print_results_to_screen, print_results_to_file, print_log_to_file, logfile);

    if(res == nullptr || res->size() == 0 || ! valid) { return Cost<DumpAll>{}; }
    auto layer_res = (*res)[0];
    if(layer_res == nullptr || layer_res->size() == 0) { return Cost<DumpAll>{}; }
    auto cluster_res = (*layer_res)[layer_res->size() - 1];

    std::map<maestro::MetricType, long double> costs;
    api->GetCostsFromAnalysisResultsSingleCluster(cluster_res, costs);

    Cost<DumpAll> cost;
    if(api->accelerator) {
      cost.valid = true;
      if constexpr (DumpAll) {
        cost.costs = costs;
        cost.costs[maestro::Area] = api->accelerator->GetArea();
      } else {
        cost.delay = costs[maestro::ExactRunTime];
        cost.energy = costs[maestro::OverallEnergy];
        cost.area = api->accelerator->GetArea();
        cost.power = api->accelerator->GetPower();
        cost.throughput = costs[maestro::Throughput];
      }
    }
    // std::cout << costs[maestro::NumUtilizedPEs] << " utilized PEs\n";
    // std::cout << costs[maestro::Throughput] << " MACs/cycle\n";
    // std::cout << costs[maestro::ExactRunTime] << " cycles\n";
    // std::cout << costs[maestro::OverallEnergy] << " x MAC energy\n";
    // if (api->accelerator) {
    //   std::cout << api->accelerator->GetArea() << " um^2\n";
    //   std::cout << api->accelerator->GetPower() << " mW\n";
    // }

    return cost;
  } catch(std::exception const & e) {
    return Cost<DumpAll>{};
  }
}

template<bool DumpAll>
Cost<DumpAll> runWrapper(int id, ShapeT const & shape, std::string const & layer_type, Point const & point, std::string const & logfile)
{
  (void) id;
  bool output_logs = false;
#ifdef _DEBUG_OUT
  output_logs = true;
#endif
  return run<DumpAll>(shape, layer_type, point, output_logs, output_logs && logfile != "", output_logs && logfile != "", logfile);
}

template<bool DumpAll>
void runBatch(
  uint64_t batch_size,
  ShapeT const & shape,
  std::string const & layer_type,
  std::vector<Point> const & points,
  std::vector<Cost<DumpAll>> & costs,
  std::ofstream & result_file,
  Point & best_point,
  Cost<DumpAll> & best_cost,
  std::string const & logfile
)
{
  assert(shape.size() != batch_size || points.size() != batch_size);

#ifdef MULTICORE
  ctpl::thread_pool pool(std::min<uint64_t>(batch_size, std::thread::hardware_concurrency()));
  std::vector<std::future<Cost<DumpAll>>> results(batch_size);

  for(uint64_t i = 0; i < batch_size; ++i) {
    Point const & point = points[i];
    results[i] = pool.push(runWrapper<DumpAll>, shape, layer_type, point, logfile);
  }

  for(uint64_t i = 0; i < batch_size; ++i) {
    costs[i] = results[i].get();
    if(result_file.is_open()) {
      printCost(result_file, costs[i]) << ',' << points[i] << '\n';
    }
    if(costs[i] < best_cost) {
      best_cost = costs[i];
      best_point = points[i];
    }
  }
#else
  bool first = true;
  Point best_point_tmp;
  Cost<DumpAll> best_cost_tmp;
  for(uint64_t i = 0; i < batch_size; ++i) {
    Point const & point = points[i];
    costs[i] = runWrapper<DumpAll>(0, shape, layer_type, point, logfile);
    if(result_file.is_open()) {
      printCost(result_file, costs[i]) << ',' << points[i] << '\n';
    }
    if(first || (costs[i] < best_cost_tmp)) {
      best_point_tmp = point;
      best_cost_tmp = costs[i];
      first = false;
    }
  }
  best_cost = best_cost_tmp;
  best_point = best_point_tmp;
#endif
}

#endif