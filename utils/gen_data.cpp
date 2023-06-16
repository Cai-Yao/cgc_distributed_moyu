#include "utils.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
using namespace std;

// 生成符合均值为 0, 标准差为 2 的随机数
normal_distribution<float> u(0, 2);
default_random_engine e(23333);

int V, E, F0, F1, F2;
string path_prefix;
string embedding_path;
string weight1_path;
string weight2_path;
namespace utils {
void init() {
  path_prefix = "data";
  embedding_path = path_prefix + "/embedding/" + to_string(V) + ".bin";
  weight1_path = path_prefix + "/weight/W1_" + to_string(F0) + "_" +
                 to_string(F1) + ".bin";
  weight2_path = path_prefix + "/weight/W2_" + to_string(F1) + "_" +
                 to_string(F2) + ".bin";
}

void gen_embedding() {
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
}

void gen_weight() {
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
}

} // namespace utils

// arg: V, E, F0, F1, F2
int main(int argc, char **argv) {
  // init
  if (argc != 6) {
    cout << "error arg number: " << argc << " is wrong!";
    return -1;
  }
  V = atoi(argv[1]);
  E = atoi(argv[2]);
  F0 = atoi(argv[3]);
  F1 = atoi(argv[4]);
  F2 = atoi(argv[5]);

  utils::init();
  utils::gen_embedding();
  utils::gen_weight();

  return 0;
}