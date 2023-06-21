#include "impls/openblas/openblas_impl.h"
#include "common.h"
#include <cblas.h>
#include <cmath>
#include <immintrin.h>

using namespace std;

namespace impl {
namespace openblas {

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

} // namespace openblas
} // namespace impl

inline float findMax(float *arr, int size) {
  __m256 maxVector = _mm256_loadu_ps(arr);
  int i = 8;
  for (; i + 7 < size; i += 8) {
    __m256 currentVector = _mm256_loadu_ps(arr + i);
    maxVector = _mm256_max_ps(maxVector, currentVector);
  }

  __m128 highVector = _mm256_extractf128_ps(maxVector, 1);
  __m128 maxValues = _mm_max_ps(_mm256_castps256_ps128(maxVector), highVector);
  for (int j = 0; j < 3; j++) {
    maxValues = _mm_max_ps(maxValues, _mm_shuffle_ps(maxValues, maxValues,
                                                     _MM_SHUFFLE(1, 0, 3, 2)));
  }

  float result = _mm_cvtss_f32(maxValues);
  for (; i < size; i++) {
    if (arr[i] > result) {
      result = arr[i];
    }
  }
  return result;
}

void impl::openblas::readGraph(const char *fname) {
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

void impl::openblas::raw_graph_to_AdjacencyList() {

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

void impl::openblas::edgeNormalization() {
  for (int i = 0; i < v_num; i++) {
    for (int j = 0; j < edge_index[i].size(); j++) {
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
      edge_val[i].push_back(val);
    }
  }
} 

void impl::openblas::readFloat(const char *fname, float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  FILE *fp = fopen(fname, "rb");
  fread(dst, num * sizeof(float), 1, fp);
  fclose(fp);
}

void impl::openblas::initFloat(float *&dst, int num) {
  dst = (float *)malloc(num * sizeof(float));
  memset(dst, 0, num * sizeof(float));
}

void impl::openblas::XW(int in_dim, int out_dim, float *in_X, float *out_X,
                        float *W) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, v_num, out_dim, in_dim,
              1.0, in_X, in_dim, W, out_dim, 0.0, out_X, out_dim);
}

void impl::openblas::AX(int dim, float *in_X, float *out_X) {
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

void impl::openblas::ReLU(int dim, float *X) {
  const int num_elements = v_num * dim;
  const int vector_size = 16;

  for (int i = 0; i < num_elements; i += vector_size) {
    __m512 values = _mm512_loadu_ps(X + i);
    __m512 zero_vector = _mm512_setzero_ps();
    __m512 result = _mm512_max_ps(values, zero_vector);
    _mm512_storeu_ps(X + i, result);
  }
}

void impl::openblas::LogSoftmax(int dim, float *X) {
  float(*tmp_X)[dim] = (float(*)[dim])X;

  for (int i = 0; i < v_num; i++) {
    // float max = findMax(X + i * dim, dim);
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

float impl::openblas::MaxRowSum(float *X, int dim) {
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
  // return findMax(X, v_num * dim);
}

void impl::openblas::freeFloats() {
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

void impl::openblas::somePreprocessing() {
  // The graph  will be transformed into adjacency list ,you can use other data
  // structure such as CSR
  raw_graph_to_AdjacencyList();
}

float impl::openblas::openblas_impl(int feature_0, int feature_1, int feature_2,
                                    const char *graph_path,
                                    const char *embedding_path,
                                    const char *weight_1_path,
                                    const char *weight_2_path,
                                    utils::util_recorder &recorder) {
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

  // Preprocessing time should be included
  auto &preprocessing = recorder.get_preprocessinig();
  recorder.begin_record("Preprocess");
  preprocessing();
  recorder.end_record("Preprocess");

  auto &edge_norm = recorder.get_edge_norm();
  recorder.begin_record("Edge Norm");
  edge_norm();
  recorder.end_record("Edge Norm");

  // printf("Layer1 XW\n");
  auto &xw_func = recorder.get_XW();
  recorder.begin_record("Layer1 XW");
  xw_func(F0, F1, X0, X1_inter, W1);
  recorder.end_record("Layer1 XW");

  // printf("Layer1 AX\n");
  auto &ax_func = recorder.get_AX();
  recorder.begin_record("Layer1 AX");
  ax_func(F1, X1_inter, X1);
  recorder.end_record("Layer1 AX");

  // printf("Layer1 ReLU\n");
  auto &relu_func = recorder.get_ReLU();
  recorder.begin_record("ReLU");
  relu_func(F1, X1);
  recorder.end_record("ReLU");

  // printf("Layer2 XW\n");
  recorder.begin_record("Layer2 XW");
  xw_func(F1, F2, X1, X2_inter, W2);
  recorder.end_record("Layer2 XW");

  // printf("Layer2 AX\n");
  recorder.begin_record("Layer2 AX");
  ax_func(F2, X2_inter, X2);
  recorder.end_record("Layer2 AX");

  // printf("Layer2 LogSoftmax\n");
  auto &logsoftmax_func = recorder.get_LogSoftmax();
  recorder.begin_record("LogSoftmax");
  logsoftmax_func(F2, X2);
  recorder.end_record("LogSoftmax");

  // You need to compute the max row sum for result verification
  auto &maxsum_func = recorder.get_MaxRowSum();
  recorder.begin_record("MaxRowSum");
  float max_sum = maxsum_func(X2, F2);
  recorder.end_record("MaxRowSum");

  // Remember to free your allocated memory
  freeFloats();

  return max_sum;
}