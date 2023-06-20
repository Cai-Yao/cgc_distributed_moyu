#ifndef GCN_IMPLS_OPENBLAS_OPENBLAS_IMPL_H_
#define GCN_IMPLS_OPENBLAS_OPENBLAS_IMPL_H_

#include "utils/utils.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace impl {
namespace openblas {

void readGraph(const char *fname);

void raw_graph_to_AdjacencyList();

void edgeNormalization();

void readFloat(const char *fname, float *&dst, int num);

void initFloat(float *&dst, int num);

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W);

void AX(int dim, float *in_X, float *out_X);

void ReLU(int dim, float *X);

void LogSoftmax(int dim, float *X);

float MaxRowSum(float *X, int dim);

void freeFloats();
void somePreprocessing();

float openblas_impl(int feature_0, int feature_1, int feature_2,
                    const char *graph_path, const char *embedding_path,
                    const char *weight_1_path, const char *weight_2_path,
                    utils::time_recorder &recorder);

} // namespace openblas
} // namespace impl

#endif