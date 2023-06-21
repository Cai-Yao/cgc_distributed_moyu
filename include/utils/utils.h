#ifndef GCN_UTILS_UTILS_H_
#define GCN_UTILS_UTILS_H_

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {
void init();
void gen_embedding();
void gen_weight();

void gen_graph(int V_, int E_);

void prepare_data(int V_, int E_, int F0_, int F1_, int F2_);

class util_recorder {
public:
  void begin_record(std::string key);
  void end_record(std::string key);
  double get_duration(std::string key);
  double get_average_duration(std::string key);
  void record_once();
  std::vector<std::string> &get_ids();
  void enable_record_time(bool enable) { enable_time = enable; }

  std::function<void(void)> &get_preprocessinig() { return preprocessing; }
  void set_preprocessing(std::function<void(void)> func) {
    preprocessing = func;
  }

  std::function<void(void)> &get_edge_norm() { return edge_norm; }
  void set_edge_norm(std::function<void(void)> func) { edge_norm = func; }

  std::function<void(int, int, float *, float *, float *)> &get_XW() {
    return XW;
  }
  void set_XW(std::function<void(int, int, float *, float *, float *)> func) {
    XW = func;
  }

  std::function<void(int, float *, float *)> &get_AX() { return AX; }
  void set_AX(std::function<void(int, float *, float *)> func) { AX = func; }

  std::function<void(int, float *)> &get_ReLU() { return ReLU; }
  void set_ReLU(std::function<void(int, float *)> func) { ReLU = func; }

  std::function<void(int, float *)> &get_LogSoftmax() { return LogSoftmax; }
  void set_LogSoftmax(std::function<void(int, float *)> func) {
    LogSoftmax = func;
  }

  std::function<float(float *, int)> &get_MaxRowSum() { return MaxRowSum; }
  void set_MaxRowSum(std::function<float(float *, int)> func) {
    MaxRowSum = func;
  }

private:
  typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
  std::unordered_map<std::string, std::pair<TimePoint, TimePoint>> hash;
  std::vector<std::string> ids;
  std::vector<std::unordered_map<std::string, std::pair<TimePoint, TimePoint>>>
      historys;
  bool enable_time;

  std::function<void(void)> preprocessing;
  std::function<void(void)> edge_norm;
  std::function<void(int, int, float *, float *, float *)> XW;
  std::function<void(int, float *, float *)> AX;
  std::function<void(int, float *)> ReLU;
  std::function<void(int, float *)> LogSoftmax;
  std::function<float(float *, int)> MaxRowSum;
};
} // namespace utils

#endif