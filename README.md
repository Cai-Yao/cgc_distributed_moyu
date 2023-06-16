## 1. 数据规模
![](./asserts/limit.jpg)

## 2. 实验
### 2.1 实验配置 
8核 15G Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
### 2.2 O3 优化
![](./asserts/O3.jpg)
### 2.3 omp+O3 优化
![](./asserts/omp_O3.jpg)

O3 大概优化4倍，再加上omp可以优化10倍以上

## 3. TODO
### 3.1 运行omp优化时，观察到cpu线程没有跑满，点和边的规模要比feature的维度大得多，这部分可以想办法优化
### 3.2 omp好像可以支持simd https://blog.csdn.net/10km/article/details/84579465
### 3.3 尝试手写多线程并发，细粒度优化