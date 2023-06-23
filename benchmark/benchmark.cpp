#include "impls/openblas/openblas_impl.h"
#include "impls/origin/origin_impl.h"
#include "utils/utils.h"
#include <benchmark/benchmark.h>
#include <string>
#include <unordered_set>
#include <vector>

std::unordered_set<std::string> data_gened;

void gen_data(int &V, int &E, int &F0, int &F1, int &F2) {
  std::string key = std::to_string(V) + "_" + std::to_string(E) + "_" +
                    std::to_string(F0) + "_" + std::to_string(F1) + "_" +
                    std::to_string(F2);
  if (data_gened.count(key)) {
    return;
  }
  utils::prepare_data(V, E, F0, F1, F2);
  data_gened.insert(key);
}

static std::string embedding_path, weight1_path, weight2_path, graph_path;
void prepare_file_path(int &V, int &E, int &F0, int &F1, int &F2) {
  std::string path_prefix = "data";
  embedding_path = path_prefix + "/embedding/" + std::to_string(V) + "_" +
                   std::to_string(F0) + ".bin";
  weight1_path = path_prefix + "/weight/W1_" + std::to_string(F0) + "_" +
                 std::to_string(F1) + ".bin";
  weight2_path = path_prefix + "/weight/W2_" + std::to_string(F1) + "_" +
                 std::to_string(F2) + ".bin";
  graph_path = path_prefix + "/graph/graph_" + std::to_string(V) + "_" +
               std::to_string(E) + ".txt";
}

static void GenSmallTestParams(benchmark::internal::Benchmark *b) {
  std::vector<int> graph_size = {4096, 16384, 65536};
  std::vector<int> f0_size = {64, 128};
  std::vector<int> f1_size = {16};
  std::vector<int> f2_size = {32};
  for (int i = 0; i < graph_size.size(); ++i) {
    for (int j = i; j < graph_size.size(); ++j) {
      for (auto &f0 : f0_size) {
        for (auto &f1 : f1_size) {
          for (auto &f2 : f2_size) {
            b->Args({graph_size[i], graph_size[j], f0, f1, f2});
          }
        }
      }
    }
  }
}

static void GenStandardTestParams(benchmark::internal::Benchmark *b) {
  std::vector<int> graph_size = {400000, 800000, 4000000};
  std::vector<int> f0_size = {128};
  std::vector<int> f1_size = {16};
  std::vector<int> f2_size = {32};
  for (int i = 0; i < graph_size.size(); ++i) {
    for (int j = i; j < graph_size.size(); ++j) {
      for (auto &f0 : f0_size) {
        for (auto &f1 : f1_size) {
          for (auto &f2 : f2_size) {
            b->Args({graph_size[i], graph_size[j], f0, f1, f2});
          }
        }
      }
    }
  }
}

utils::util_recorder *recorder;

static void BM_OriginImpl(benchmark::State &state) {
  int V = state.range(0), E = state.range(1), F0 = state.range(2),
      F1 = state.range(3), F2 = state.range(4);
  gen_data(V, E, F0, F1, F2);
  prepare_file_path(V, E, F0, F1, F2);
  auto standard_res = impl::origin::origin_impl(
      F0, F1, F2, graph_path.c_str(), embedding_path.c_str(),
      weight1_path.c_str(), weight2_path.c_str(), *recorder);
  float diff = 0;
  double run_time = 0;
  recorder->enable_record_time(true);
  for (auto _ : state) {
    auto case_res = impl::origin::origin_impl(
        F0, F1, F2, graph_path.c_str(), embedding_path.c_str(),
        weight1_path.c_str(), weight2_path.c_str(), *recorder);

    diff = std::max(diff, abs(standard_res - case_res));
    run_time = 0;
    for (auto &id : recorder->get_ids()) {
      run_time += recorder->get_duration(id);
    }
    state.SetIterationTime(run_time / 1e3);
    recorder->record_once();
  }
  state.counters["Max Diff"] = diff;
  for (auto &id : recorder->get_ids()) {
    state.counters[id] = recorder->get_average_duration(id);
  }
}

static void BM_OpenBlasImpl(benchmark::State &state) {
  int V = state.range(0), E = state.range(1), F0 = state.range(2),
      F1 = state.range(3), F2 = state.range(4);
  gen_data(V, E, F0, F1, F2);
  prepare_file_path(V, E, F0, F1, F2);
  auto standard_res = impl::origin::origin_impl(
      F0, F1, F2, graph_path.c_str(), embedding_path.c_str(),
      weight1_path.c_str(), weight2_path.c_str(), *recorder);
  float diff = 0;
  double run_time = 0;
  recorder->enable_record_time(true);
  for (auto _ : state) {
    auto case_res = impl::openblas::openblas_impl(
        F0, F1, F2, graph_path.c_str(), embedding_path.c_str(),
        weight1_path.c_str(), weight2_path.c_str(), *recorder);

    diff = std::max(diff, abs(standard_res - case_res));
    run_time = 0;
    for (auto &id : recorder->get_ids()) {
      run_time += recorder->get_duration(id);
    }
    state.SetIterationTime(run_time / 1e3);
    recorder->record_once();
  }
  state.counters["Max Diff"] = diff;
  for (auto &id : recorder->get_ids()) {
    state.counters[id] = recorder->get_average_duration(id);
  }
}

static void BaseSetup(const benchmark::State &state) {
  recorder = new utils::util_recorder();
}

static void BaseTeardown(const benchmark::State &state) { delete recorder; }

static void OpenBlasBaseSetup(const benchmark::State &state) {
  recorder = new utils::util_recorder();
  recorder->set_preprocessing(impl::openblas::somePreprocessing);
  recorder->set_edge_norm(impl::openblas::edgeNormalization);
  recorder->set_XW(impl::openblas::XW);
  recorder->set_AX(impl::openblas::AX);
  recorder->set_ReLU(impl::openblas::ReLU);
  recorder->set_LogSoftmax(impl::openblas::LogSoftmax);
  recorder->set_MaxRowSum(impl::openblas::MaxRowSum);
}

static void OpenBlasOptSetup(const benchmark::State &state) {
  recorder = new utils::util_recorder();
  recorder->set_preprocessing(impl::openblas::somePreprocessing);
  recorder->set_edge_norm(impl::openblas::edgeNormalization);
  recorder->set_XW(impl::openblas::XW);
  recorder->set_AX(impl::openblas::AX);
  recorder->set_ReLU(impl::openblas::ReLU);
  recorder->set_LogSoftmax(impl::openblas::LogSoftmaxOpt);
  recorder->set_MaxRowSum(impl::openblas::MaxRowSumOpt);
}

BENCHMARK(BM_OriginImpl)
    ->Name("Origin Implemention Small")
    ->Apply(GenSmallTestParams)
    ->Setup(BaseSetup)
    ->Teardown(BaseTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_OpenBlasImpl)
    ->Name("OpenBlas Implemention Small")
    ->Apply(GenSmallTestParams)
    ->Setup(OpenBlasBaseSetup)
    ->Teardown(BaseTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_OriginImpl)
    ->Name("Origin Implemention Standard")
    ->Apply(GenStandardTestParams)
    ->Setup(BaseSetup)
    ->Teardown(BaseTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(5);

BENCHMARK(BM_OpenBlasImpl)
    ->Name("OpenBlas Implemention Standard")
    ->Apply(GenStandardTestParams)
    ->Setup(OpenBlasBaseSetup)
    ->Teardown(BaseTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(5);

BENCHMARK(BM_OpenBlasImpl)
    ->Name("OpenBlas Implemention Standard Opt")
    ->Apply(GenStandardTestParams)
    ->Setup(OpenBlasOptSetup)
    ->Teardown(BaseTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(5);

BENCHMARK_MAIN();