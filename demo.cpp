#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <random>
#include <ctime>
#include <new>
#include <vector>

using namespace std;

// 生成符合均值为 0, 标准差为 2 的随机数
normal_distribution<float> u(0, 2);
default_random_engine e(23333);

float *arr;

void init(int n) {
    arr = new float[n];
    for (int i = 0; i < n; i++) {
        arr[i] = u(e);
    }
}

void output(float* X, int n) {
  for (int i = 0; i < n; i++) {
    cout << X[i] << " ";
  }
  cout << endl;
}

float findMax(float* arr, int size) {
    __m256 maxVector = _mm256_loadu_ps(arr); // 加载前8个元素到AVX向量寄存器
    int i = 8;
    for (; i + 7 < size; i += 8) {
        __m256 currentVector = _mm256_loadu_ps(arr + i); // 加载下一个8个元素到AVX向量寄存器
        maxVector = _mm256_max_ps(maxVector, currentVector); // 找到最大值向量
    }

    // 在向量寄存器内部找到最大值
    __m128 highVector = _mm256_extractf128_ps(maxVector, 1);
    __m128 maxValues = _mm_max_ps(_mm256_castps256_ps128(maxVector), highVector);
    for (int j = 0; j < 3; j++) {
        maxValues = _mm_max_ps(maxValues, _mm_shuffle_ps(maxValues, maxValues, _MM_SHUFFLE(1, 0, 3, 2)));
    }

    // 在最大值向量中找到最大值
    float result = _mm_cvtss_f32(maxValues);
    for (; i < size; i++) {
        if (arr[i] > result) {
            result = arr[i];
        }
    }
    return result;
}

const int v_num = 10;
void ReLU(int dim, float *X) {
    const int num_elements = v_num * dim;
    const int vector_size = 8;
    const int num_threads = 4;

// #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_elements; i += vector_size) {
        __m256 values = _mm256_loadu_ps(X + i);
        __m256 zero_vector = _mm256_setzero_ps();
        __m256 result = _mm256_max_ps(values, zero_vector);
        _mm256_storeu_ps(X + i, result);
    }
}

int main()
{
    // const int N = 12345;
    // clock_t start,end;
    // double endtime;
    
    // init(N);
    // findMax(arr, N);
    // init(5*N);
    // findMax(arr, 5*N);
    vector<int> t;
    t.resize(10);
    t.push_back(100);
    t.push_back(101);
    for (int i = 0; i < t.size(); i++) {
        cout << t[i] << " ";
    }
    cout << endl;
}