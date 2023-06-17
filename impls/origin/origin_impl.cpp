#include "impls/origin/origin_impl.h"

using namespace std;

namespace impl {
namespace origin {

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

} // namespace origin
} // namespace impl

void impl::origin::readGraph(const char *fname) {
  ifstream infile(fname);

  int source;
  int end;

  infile >> v_num >> e_num;

  raw_graph.resize(e_num * 2);

  while (!infile.eof()) {
    infile >> source >> end;
    if (infile.peek() == EOF)
      break;
    raw_graph.push_back(source);
    raw_graph.push_back(end);
  }
}

void impl::origin::raw_graph_to_AdjacencyList() {

  int src;
  int dst;

  edge_index.resize(v_num);
  edge_val.resize(v_num);
  degree.resize(v_num, 0);

  for (int i = 0; i < raw_graph.size() / 2; i++) {
    src = raw_graph[2 * i];
    dst = raw_graph[2 * i + 1];
    edge_index[dst].push_back(src);
    degree[src]++;
  }
}

void impl::origin::edgeNormalization() {
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < edge_index[i].size(); j++) {
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
      edge_val[i].push_back(val);
    }
  }
}

void impl::origin::readFloat(const char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void impl::origin::initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

void impl::origin::XW(int in_dim, int out_dim, float *in_X, float *out_X,
                      float *W) {
  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
  float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < out_dim; j++) {
      for (int k = 0; k < in_dim; k++) {
        tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
      }
    }
  }
}

void impl::origin::AX(int dim, float *in_X, float *out_X) {
  float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
  float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

  for (int i = 0; i < v_num; i++) {
    vector<int> &nlist = edge_index[i];
    for (int j = 0; j < nlist.size(); j++) {
      int nbr = nlist[j];
      for (int k = 0; k < dim; k++) {
        tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
      }
    }
  }
}

void impl::origin::ReLU(int dim, float *X) {
  for (int i = 0; i < v_num * dim; i++)
    if (X[i] < 0)
      X[i] = 0;
}

void impl::origin::LogSoftmax(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

  for (int i = 0; i < v_num; i++) {
    float max = tmp_X[i][0];
    for (int j = 1; j < dim; j++) {
      if (tmp_X[i][j] > max)
        max = tmp_X[i][j];
    }

    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += exp(tmp_X[i][j] - max);
    }
    sum = log(sum);

    for (int j = 0; j < dim; j++) {
      tmp_X[i][j] = tmp_X[i][j] - max - sum;
    }
  }
}

float impl::origin::MaxRowSum(float *X, int dim) {
  float(*tmp_X)[dim] = (float(*)[dim])X;
  float max = -__FLT_MAX__;

  for (int i = 0; i < v_num; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++) {
      sum += tmp_X[i][j];
    }
    if (sum > max)
      max = sum;
  }
  return max;
}

void impl::origin::freeFloats() {
  free(X0);
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
  edge_index.clear();
  edge_val.clear();
  degree.clear();
  raw_graph.clear();
}

void impl::origin::somePreprocessing() {
  // The graph  will be transformed into adjacency list ,you can use other data
  // structure such as CSR
  raw_graph_to_AdjacencyList();
}

pair<float, double> impl::origin::origin_impl(int feature_0, int feature_1,
                                              int feature_2,
                                              const char *graph_path,
                                              const char *embedding_path,
                                              const char *weight_1_path,
                                              const char *weight_2_path) {
  int F0 = 0, F1 = 0, F2 = 0;

  F0 = feature_0;
  F1 = feature_1;
  F2 = feature_2;

  readGraph(graph_path);
  readFloat(embedding_path, X0, v_num * F0);
  readFloat(weight_1_path, W1, F0 * F1);
  readFloat(weight_2_path, W2, F1 * F2);

  initFloat(X1, v_num * F1);
  initFloat(X1_inter, v_num * F1);
  initFloat(X2, v_num * F2);
  initFloat(X2_inter, v_num * F2);

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();

  // Preprocessing time should be included

  somePreprocessing();

  edgeNormalization();

  // printf("Layer1 XW\n");
  XW(F0, F1, X0, X1_inter, W1);

  // printf("Layer1 AX\n");
  AX(F1, X1_inter, X1);

  // printf("Layer1 ReLU\n");
  ReLU(F1, X1);

  // printf("Layer2 XW\n");
  XW(F1, F2, X1, X2_inter, W2);

  // printf("Layer2 AX\n");
  AX(F2, X2_inter, X2);

  // printf("Layer2 LogSoftmax\n");
  LogSoftmax(F2, X2);

  // You need to compute the max row sum for result verification
  float max_sum = MaxRowSum(X2, F2);

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now();
  chrono::duration<double> l_durationSec = end - start;
  double l_timeMs = l_durationSec.count() * 1e3;

  // Remember to free your allocated memory
  freeFloats();

  return {max_sum, l_timeMs};
}