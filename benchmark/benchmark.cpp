#include "impls/origin/origin_impl.h"
#include "utils/utils.h"
#include <benchmark/benchmark.h>
#include <string>
#include <unordered_set>
#include <vector>

std::unordered_set<std::string> data_gened;

void gen_data(int V, int E, int F0, int F1, int F2) {
  std::string key = std::to_string(V) + "_" + std::to_string(E) + "_" +
                    std::to_string(F0) + "_" + std::to_string(F1) + "_" +
                    std::to_string(F2);
  if (data_gened.count(key)) {
    return;
  }
  utils::prepare_data(V, E, F0, F1, F2);
  data_gened.insert(key);
}

static void BM_OriginImpl(benchmark::State &state) {}

BENCHMARK_MAIN();