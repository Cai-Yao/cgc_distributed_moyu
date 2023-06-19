#ifndef GCN_UTILS_UTILS_H_
#define GCN_UTILS_UTILS_H_

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {
void init();
void gen_embedding();
void gen_weight();

void gen_graph(int V_, int E_);

void prepare_data(int V_, int E_, int F0_, int F1_, int F2_);

class time_recorder {
public:
  void begin_record(std::string key);
  void end_record(std::string key);
  double get_duration(std::string key);
  double get_average_duration(std::string key);
  void record_once();
  std::vector<std::string> &get_ids();

private:
  typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
  std::unordered_map<std::string, std::pair<TimePoint, TimePoint>> hash;
  std::vector<std::string> ids;
  std::vector<std::unordered_map<std::string, std::pair<TimePoint, TimePoint>>>
      historys;
};
} // namespace utils

#endif