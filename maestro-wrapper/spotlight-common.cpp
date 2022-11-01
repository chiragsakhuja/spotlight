#include "spotlight-common.hpp"

std::string MetricTypeToString(maestro::MetricType const & t)
{
  using namespace maestro;

  static const std::map<MetricType, char const *> convert {
    { Computations, "Computations" },
    { AbsComputations, "AbsComputations" },
    { ExactRunTime, "ExactRunTime" },
    { MaxRunTime, "MaxRunTime" },
    { Throughput, "Throughput" },
    { ThroughputMin, "ThroughputMin" },
    { ThroughputMax, "ThroughputMax" },
    { AbsThroughput, "AbsThroughput" },
    { AbsThroughputMin, "AbsThroughputMin" },
    { AbsThroughputMax, "AbsThroughputMax" },
    { InputL2BufferReq, "InputL2BufferReq" },
    { InputL1BufferReq, "InputL1BufferReq" },
    { InputL2BufferWrite, "InputL2BufferWrite" },
    { InputL2BufferRead, "InputL2BufferRead" },
    { InputL1BufferWrite, "InputL1BufferWrite" },
    { InputL1BufferRead, "InputL1BufferRead" },
    { InputReuseFactor, "InputReuseFactor" },
    { FilterL2BufferReq, "FilterL2BufferReq" },
    { FilterL1BufferReq, "FilterL1BufferReq" },
    { FilterL2BufferWrite, "FilterL2BufferWrite" },
    { FilterL2BufferRead, "FilterL2BufferRead" },
    { FilterL1BufferWrite, "FilterL1BufferWrite" },
    { FilterL1BufferRead, "FilterL1BufferRead" },
    { FilterReuseFactor, "FilterReuseFactor" },
    { OutputL2BufferReq, "OutputL2BufferReq" },
    { OutputL1BufferReq, "OutputL1BufferReq" },
    { OutputL2BufferWrite, "OutputL2BufferWrite" },
    { OutputL2BufferRead, "OutputL2BufferRead" },
    { OutputL1BufferWrite, "OutputL1BufferWrite" },
    { OutputL1BufferRead, "OutputL1BufferRead" },
    { OutputReuseFactor, "OutputReuseFactor" },
    { OverallReuseFactor, "OverallReuseFactor" },
    { InputL2BufferWriteEnergy, "InputL2BufferWriteEnergy" },
    { InputL2BufferReadEnergy, "InputL2BufferReadEnergy" },
    { InputL1BufferWriteEnergy, "InputL1BufferWriteEnergy" },
    { InputL1BufferReadEnergy, "InputL1BufferReadEnergy" },
    { FilterL2BufferWriteEnergy, "FilterL2BufferWriteEnergy" },
    { FilterL2BufferReadEnergy, "FilterL2BufferReadEnergy" },
    { FilterL1BufferWriteEnergy, "FilterL1BufferWriteEnergy" },
    { FilterL1BufferReadEnergy, "FilterL1BufferReadEnergy" },
    { OutputL2BufferWriteEnergy, "OutputL2BufferWriteEnergy" },
    { OutputL2BufferReadEnergy, "OutputL2BufferReadEnergy" },
    { OutputL1BufferWriteEnergy, "OutputL1BufferWriteEnergy" },
    { OutputL1BufferReadEnergy, "OutputL1BufferReadEnergy" },
    { OverallL2WriteEnergy, "OverallL2WriteEnergy" },
    { OverallL2ReadEnergy, "OverallL2ReadEnergy" },
    { OverallL1WriteEnergy, "OverallL1WriteEnergy" },
    { OverallL1ReadEnergy, "OverallL1ReadEnergy" },
    { OverallEnergy, "OverallEnergy" },
    { PeakBWReq, "PeakBWReq" },
    { AvgBWReq, "AvgBWReq" },
    { IngressDelayMin, "IngressDelayMin" },
    { IngressDelayMax, "IngressDelayMax" },
    { IngressDelayAvg, "IngressDelayAvg" },
    { EgressDelayMin, "EgressDelayMin" },
    { EgressDelayMax, "EgressDelayMax" },
    { EgressDelayAvg, "EgressDelayAvg" },
    { ComputationDelayMin, "ComputationDelayMin" },
    { ComputationDelayMax, "ComputationDelayMax" },
    { ComputationDelayAvg, "ComputationDelayAvg" },
    { NumUtilizedPEs, "NumUtilizedPEs" },
    { Area, "Area" }
  };

  auto it = convert.find(t);
  return it == convert.end() ? "Invalid" : std::string(it->second);
}


std::ostream & operator<<(std::ostream & out, Point const & point)
{
#ifdef _DEBUG_OUT
  out << "# PEs: " << point.num_pes << '\n';
  out << "# SIMD Lanes: " << point.num_simd_lanes << '\n';
  out << "L1 Size: " << point.l1_size << '\n';
  out << "L2 Size: " << point.l2_size << '\n';
  out << "Bit Width: " << point.bit_width << '\n';
  out << "BW: " << point.bw << '\n';
  out << "Latency: " << point.latency << '\n';
  for(auto const & elem : point.dataflow) {
    char type = std::get<0>(elem);
    uint64_t extent = std::get<1>(elem);
    std::string var = std::get<2>(elem);
    out << type << ' ' << extent << ' ' << var << '\n';
  }
#else
  for(auto const & elem : point.dataflow) {
    char type = std::get<0>(elem);
    uint64_t extent = std::get<1>(elem);
    std::string var = std::get<2>(elem);
    out << type << ' ' << extent << ' ' << var << " | ";
  }
#endif
  return out;
}

std::shared_ptr<maestro::ConfigurationV2> setupConfig(uint64_t num_pe, uint64_t num_simd_lanes, uint64_t l1_size, uint64_t l2_size, uint64_t bit_width, uint64_t bw, uint64_t offchip_bw, uint64_t hops)
{
  auto noc_multcast = std::make_shared<std::vector<bool>>();
  auto noc_latency = std::make_shared<std::vector<int>>();
  auto noc_bw = std::make_shared<std::vector<int>>();
  noc_bw->push_back(bw);
  noc_bw->push_back(bw);
  noc_bw->push_back(bw);
  noc_bw->push_back(bw);
  noc_latency->push_back(hops);
  noc_latency->push_back(hops);
  noc_latency->push_back(hops);
  noc_latency->push_back(hops);
  noc_multcast->push_back(true);
  noc_multcast->push_back(true);
  noc_multcast->push_back(true);
  noc_multcast->push_back(true);

  auto config = std::make_shared<maestro::ConfigurationV2>("", "", bit_width, noc_bw, noc_latency, noc_multcast, num_pe, num_simd_lanes, bw, l1_size, l2_size, offchip_bw);

  return config;
}

void setupDFSL(std::shared_ptr<maestro::ConfigurationV2> config, ShapeT const & shape, std::string const & layer_type, DataflowT const & dataflow)
{
  using namespace maestro;

  auto dim_vector = std::make_shared<std::vector<std::shared_ptr<DFA::LayerDimension>>>();

  for(auto const & elem : shape) {
    std::string dim(1, elem.first);
    dim_vector->push_back(std::make_shared<DFA::LayerDimension>(dim, elem.second.first, elem.second.second, 1));
  }

  auto directive_table = std::make_shared<DFA::DirectiveTable>();
  for(auto const & elem : dataflow) {
    char type = std::get<0>(elem);
    uint64_t extent = std::get<1>(elem);
    std::string const & var = std::get<2>(elem);

    if (type == 'S') {
      directive_table->AddDirective(std::make_shared<DFA::directive::SpatialMap>(extent, extent, var));
    } else if (type == 'T') {
      if (var == "X" || var == "Y") {
        directive_table->AddDirective(
            std::make_shared<DFA::directive::TemporalMap>(extent, 1, var));
      } else {
        directive_table->AddDirective(
            std::make_shared<DFA::directive::TemporalMap>(extent, extent, var));
      }
    } else if (type == 'C') {
      if (var == "P") {
        directive_table->AddDirective(std::make_shared<DFA::directive::Cluster>(extent, DFA::directive::ClusterType::Physical));
      } else if (var == "L") {
        directive_table->AddDirective(std::make_shared<DFA::directive::Cluster>(extent, DFA::directive::ClusterType::Logical));
      }
    }
  }

  if(layer_type == "CONV") {
    auto curr_layer = std::make_shared<DFA::ConvLayer>("Conv");
    curr_layer->SetDimensions(dim_vector);
    curr_layer->SetDataflow(directive_table);
    curr_layer->SetLayerType(LayerType::CONV);
    config->network_->SetName("VTA");
    config->network_->AddLayer(curr_layer);

#ifdef _DEBUG_OUT
    std::cout << "\n=== MAESTRO Setup Begin\n";
    std::cout << curr_layer->ToString();
    std::cout << "=== MAESTRO Setup End\n\n";
#endif
  } else if(layer_type == "DSCONV") {
    auto curr_layer = std::make_shared<DFA::DSConvLayer>("DSConv");
    curr_layer->SetDimensions(dim_vector);
    curr_layer->SetDataflow(directive_table);
    curr_layer->SetLayerType(LayerType::DSCONV);
    config->network_->SetName("VTA");
    config->network_->AddLayer(curr_layer);

#ifdef _DEBUG_OUT
    std::cout << "\n=== MAESTRO Setup Begin\n";
    std::cout << curr_layer->ToString();
    std::cout << "=== MAESTRO Setup End\n\n";
#endif
  }


}

std::shared_ptr<maestro::APIV2> configure(uint64_t num_pe, uint64_t num_simd_lanes, uint64_t l1_size, uint64_t l2_size, uint64_t bit_width, uint64_t bw, uint64_t offchip_bw, uint64_t hops, ShapeT const & shape, std::string const & layer_type, DataflowT const & dataflow)
{
  auto config = setupConfig(num_pe, num_simd_lanes, l1_size, l2_size, bit_width, bw, offchip_bw, hops);
  setupDFSL(config, shape, layer_type, dataflow);

  std::shared_ptr<maestro::APIV2> api = std::make_shared<maestro::APIV2>(config, false);

  return api;
}
