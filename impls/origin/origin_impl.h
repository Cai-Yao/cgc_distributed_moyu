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

void readGraph(char *fname);

void raw_graph_to_AdjacencyList();

void edgeNormalization();

void readFloat(char *fname, float *&dst, int num);

void initFloat(float *&dst, int num);

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W);

void AX(int dim, float *in_X, float *out_X);

void ReLU(int dim, float *X);

void LogSoftmax(int dim, float *X);

float MaxRowSum(float *X, int dim);

void freeFloats();
void somePreprocessing();

std::pair<float, double> origin_impl(int feature_0, int feature_1,
                                     int feature_2, char *graph_path,
                                     char *embedding_path, char *weight_1_path,
                                     char *weight_2_path);