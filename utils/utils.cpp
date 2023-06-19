#include "utils/utils.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
using namespace std;

namespace utils {
// 生成符合均值为 0, 标准差为 2 的随机数
normal_distribution<float> u(0, 2);
default_random_engine e(23333);

int V, E, F0, F1, F2;
string path_prefix;
string embedding_path;
string weight1_path;
string weight2_path;

unordered_set<string> gened;

void init() {
  path_prefix = "data";
  embedding_path =
      path_prefix + "/embedding/" + to_string(V) + "_" + to_string(F0) + ".bin";
  weight1_path = path_prefix + "/weight/W1_" + to_string(F0) + "_" +
                 to_string(F1) + ".bin";
  weight2_path = path_prefix + "/weight/W2_" + to_string(F1) + "_" +
                 to_string(F2) + ".bin";
}

void gen_embedding() {
  if (gened.count(embedding_path)) {
    return;
  }

  ofstream ofile;
  ofile.open(embedding_path, ios::binary);
  if (!ofile) {
    cout << embedding_path << " is not exist" << endl;
    return;
  }

  for (int i = 0; i < V; i++) {
    for (int j = 0; j < F0; j++) {
      float f = u(e);
      ofile.write((char *)&f, sizeof(f));
    }
  }

  ofile.close();
  gened.insert(embedding_path);
}

void gen_weight() {
  if (gened.count(weight1_path) && gened.count(weight2_path)) {
    return;
  }
  ofstream ofile1;
  ofstream ofile2;
  ofile1.open(weight1_path, ios::binary);
  ofile2.open(weight2_path, ios::binary);
  if (!ofile1) {
    cout << weight1_path << " is not exist" << endl;
    return;
  }
  if (!ofile2) {
    cout << weight2_path << " is not exist" << endl;
    return;
  }

  for (int i = 0; i < F0; i++) {
    for (int j = 0; j < F1; j++) {
      float f = u(e);
      ofile1.write((char *)&f, sizeof(f));
    }
  }
  for (int i = 0; i < F1; i++) {
    for (int j = 0; j < F2; j++) {
      float f = u(e);
      ofile2.write((char *)&f, sizeof(f));
    }
  }

  ofile1.close();
  ofile2.close();
  gened.insert(weight1_path);
  gened.insert(weight2_path);
}

void gen_graph(int V_, int E_) {
  string file_name =
      "data/graph/graph_" + to_string(V_) + "_" + to_string(E_) + ".txt";

  if (gened.count(file_name)) {
    return;
  }
  string cmd = "./PaRMAT -nVertices " + to_string(V_) + " -nEdges " +
               to_string(E_) + " -output " + file_name + " > /dev/null";
  int res = system(cmd.c_str());
  string sed_cmd =
      "sed -i \"1i" + to_string(V_) + " " + to_string(E_) + "\" " + file_name;
  res = system(sed_cmd.c_str());
  gened.insert(file_name);
  return;
}

void prepare_data(int V_, int E_, int F0_, int F1_, int F2_) {
  V = V_;
  E = E_;
  F0 = F0_;
  F1 = F1_;
  F2 = F2_;

  init();
  gen_embedding();
  gen_weight();
  gen_graph(V_, E_);
}

void time_recorder::begin_record(std::string key) {
  if (!hash.count(key)) {
    ids.push_back(key);
  }
  TimePoint now = std::chrono::steady_clock::now();
  hash[key] = {now, now};
}

void time_recorder::end_record(std::string key) {
  if (!hash.count(key)) {
    ids.push_back(key);
  }
  TimePoint now = std::chrono::steady_clock::now();
  auto &iter = hash[key];
  iter.second = now;
}

double time_recorder::get_duration(std::string key) {
  if (!hash.count(key)) {
    return 0;
  }
  auto &iter = hash[key];
  return std::chrono::duration<double, std::milli>(iter.second - iter.first)
      .count();
}

double time_recorder::get_average_duration(std::string key) {
  double cum = 0;
  int cnt = 0;
  for (auto &history : historys) {
    if (history.count(key)) {
      auto &iter = history[key];
      cum += std::chrono::duration<double, std::milli>(iter.second - iter.first)
                 .count();
      ++cnt;
    }
  }
  return cum / cnt;
}

std::vector<std::string> &time_recorder::get_ids() { return ids; }

void time_recorder::record_once() {
  historys.emplace_back(hash);
  hash.clear();
}

} // namespace utils