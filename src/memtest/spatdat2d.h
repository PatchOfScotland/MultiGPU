#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

// Ensure alignment to power of 2 due to the indexing
int64_t overalloc(int64_t n, int64_t m) {
    int64_t maxed = std::max(n, m);
    return std::pow(2, std::ceil(std::log2(maxed*maxed)));
}

template <typename T>
class Datastruct {
    protected:
        T *data;
        int64_t n, m;

    public:
        std::string name;

        Datastruct(int64_t n, int64_t m, std::string name) {
            this->n = n;
            this->m = m;
            this->name = name;
        }

        virtual T& operator()(int64_t i, int64_t j) = 0;
};

template <typename T>
class Flat : public Datastruct<T> {
    public:
        Flat(int64_t n, int64_t m) : Datastruct<T>(n, m, "Flat") {
            this->data = new T[n * m];
        }

        Flat(int64_t n, int64_t m, T *data) : Flat(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = data[i*m + j];
                }
            }
        }

        Flat(int64_t n, int64_t m, T val) : Flat(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = val;
                }
            }
        }

        ~Flat() {
            delete[] this->data;
        }

        T& operator()(int64_t i, int64_t j) override {
            return this->data[i * this->m + j];
        }
};

template <typename T>
class HilbertCurve : public Datastruct<T> {
    private:
        int64_t hilbert_index(int64_t x, int64_t y) {
            int64_t a = x ^ y;
            int64_t b = 0xFFFF ^ a;
            int64_t c = 0xFFFF ^ (x | y);
            int64_t d = x & (y ^ 0xFFFF);

            int64_t A = a | (b >> 1);
            int64_t B = (a >> 1) ^ a;
            int64_t C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
            int64_t D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;

            a = A; b = B; c = C; d = D;
            A = ((a & (a >> 2)) ^ (b & (b >> 2)));
            B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)));
            C ^= ((a & (c >> 2)) ^ (b & (d >> 2)));
            D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)));

            a = A; b = B; c = C; d = D;
            A = ((a & (a >> 4)) ^ (b & (b >> 4)));
            B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)));
            C ^= ((a & (c >> 4)) ^ (b & (d >> 4)));
            D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)));

            a = A; b = B; c = C; d = D;
            C ^= ((a & (c >> 8)) ^ (b & (d >> 8)));
            D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)));

            a = C ^ (C >> 1);
            b = D ^ (D >> 1);

            int64_t i0 = x ^ y;
            int64_t i1 = b | (0xFFFF ^ (i0 | a));

            i0 = (i0 | (i0 << 8)) & 0x00FF00FF;
            i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F;
            i0 = (i0 | (i0 << 2)) & 0x33333333;
            i0 = (i0 | (i0 << 1)) & 0x55555555;

            i1 = (i1 | (i1 << 8)) & 0x00FF00FF;
            i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F;
            i1 = (i1 | (i1 << 2)) & 0x33333333;
            i1 = (i1 | (i1 << 1)) & 0x55555555;

            return (i1 << 1) | i0;
        }

    public:
        HilbertCurve(int64_t n, int64_t m) : Datastruct<T>(n, m, "Hilbert") {
            this->data = new T[n * m];
        }

        HilbertCurve(int64_t n, int64_t m, T *data) : HilbertCurve(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = data[i * m + j];
                }
            }
        }

        HilbertCurve(int64_t n, int64_t m, T val) : HilbertCurve(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = val;
                }
            }
        }

        ~HilbertCurve() {
            delete[] this->data;
        }

        T& operator()(int64_t i, int64_t j) override {
            int64_t idx = hilbert_index(i, j);
            return this->data[idx];
        }
};

template <typename T>
class ZCurve : public Datastruct<T> {
    private:
        int64_t interleave(int64_t x, int64_t y) {
            int64_t z = 0;
            for (int64_t i = 0; i < 32; i++) {
                z |= ((y & (1 << i)) << i) | ((x & (1 << i)) << (i + 1));
            }
            return z;
        }

        int64_t part1by1(int64_t n) {
            n &= 0x0000ffff;
            n = (n | (n << 8)) & 0x00ff00ff;
            n = (n | (n << 4)) & 0x0f0f0f0f;
            n = (n | (n << 2)) & 0x33333333;
            n = (n | (n << 1)) & 0x55555555;
            return n;
        }

        int64_t interleave2(int64_t y, int64_t x) {
            return part1by1(x) | (part1by1(y) << 1);
        }

    public:
        ZCurve(int64_t n, int64_t m) : Datastruct<T>(n, m, "ZCurve") {
            int64_t size = overalloc(n, m);
            this->data = new T[size];
        }

        ZCurve(int64_t n, int64_t m, T *data) : ZCurve(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = data[i * m + j];
                }
            }
        }

        ZCurve(int64_t n, int64_t m, T val) : ZCurve(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = val;
                }
            }
        }

        ~ZCurve() {
            delete[] this->data;
        }

        T& operator()(int64_t i, int64_t j) override {
            int64_t idx = interleave2(i, j);
            return this->data[idx];
        }
};

// Implementation that uses flat indexing on the "outer" addresses, but the 
// inner addresses are interleaved using a Z-curve stored in a LUT.
template <typename T>
class ZCurveLUT : public Datastruct<T> {
    private:
        uint8_t *lut;
        static constexpr int64_t square_lut_size = 16;

        uint8_t interleave(uint8_t x, uint8_t y) {
            uint8_t z = 0;
            for (int8_t i = 0; i < square_lut_size; i++) {
                z |= ((y & (1 << i)) << i) | ((x & (1 << i)) << (i + 1));
            }
            return z;
        }

    public:
        ZCurveLUT(int64_t n, int64_t m) : Datastruct<T>(n, m, "ZCurveLUT") {
            int64_t size = overalloc(n, m);
            this->data = new T[size];
            this->lut = new uint8_t[square_lut_size*square_lut_size];
            for (int8_t i = 0; i < square_lut_size; i++) {
                for (int8_t j = 0; j < square_lut_size; j++) {
                    this->lut[i*square_lut_size + j] = interleave(i, j);
                }
            }
        }

        ZCurveLUT(int64_t n, int64_t m, T *data) : ZCurveLUT(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = data[i * m + j];
                }
            }
        }

        ZCurveLUT(int64_t n, int64_t m, T val) : ZCurveLUT(n, m) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = 0; j < m; j++) {
                    (*this)(i,j) = val;
                }
            }
        }

        ~ZCurveLUT() {
            delete[] this->data;
            delete[] this->lut;
        }

        T& operator()(int64_t i, int64_t j) override {
            constexpr int64_t
                square_lut = square_lut_size * square_lut_size,
                lower_mask = square_lut - 1,
                upper_mask = 0xFFFFFFFFFFFFFF00 | (0xFF - lower_mask);
            int64_t flat_idx = i * this->m + j;
            int64_t outer_idx = flat_idx & upper_mask;
            uint8_t inner_idx = lut[flat_idx & lower_mask];
            int64_t idx = outer_idx | inner_idx;
            return this->data[idx];
        }
};

uint64_t bench_flat_access(std::vector<float> regular, Datastruct<float> &custom, int64_t n, int64_t m) {
    auto begin = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            assert (regular[i * m + j] == custom(i, j) && "Indexing error");
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

uint64_t bench_trans_access(std::vector<float> regular, Datastruct<float> &custom, int64_t n, int64_t m) {
    auto begin = std::chrono::high_resolution_clock::now();
    for (int64_t j = 0; j < m; j++) {
        for (int64_t i = 0; i < n; i++) {
            assert (regular[i * m + j] == custom(i, j) && "Indexing error");
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

uint64_t bench_2d_sum(Datastruct<float> &custom, int64_t n, int64_t m, int64_t r) {
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<float> sums(n*m, 0.0f);
    //#pragma omp parallel for collapse(2) num_threads(8) schedule(static)
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            float sum = 0.0f;
            for (int64_t k = -r; k <= r; k++) {
                for (int64_t l = -r; l <= r; l++) {
                    if (i + k >= 0 && i + k < n && j + l >= 0 && j + l < m) {
                        sum += custom(i + k, j + l);
                    }
                }
            }
            sums[i * m + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

void pretty_print(std::string benchmark, std::vector<std::string> names, std::vector<uint64_t> times) {
    assert (names.size() == times.size() && "Names and times must be of equal length");

    int64_t max_name_length = 0;
    for (auto name : names) {
        max_name_length = std::max(max_name_length, static_cast<int64_t>(name.size()));
    }
    int64_t max_time_length = 0;
    for (auto time : times) {
        max_time_length = std::max(max_time_length, static_cast<int64_t>(std::to_string(time).size()));
    }

    std::cout << "Benchmarking " << benchmark << std::endl;
    for (int64_t i = 0; i < names.size(); i++) {
        std::cout
            << names[i] << std::string(max_name_length - names[i].size(), ' ') << " : " << std::string(max_time_length - std::to_string(times[i]).size(), ' ') << times[i] << " ns" << std::endl;
    }
    std::cout << "-------------------" << std::endl;
}

//int main() {
//    const int64_t n = 512;
//    std::vector<float> data(n*n, 0.0f);
//    for (int64_t i = 0; i < n*n; i++) {
//        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//    }
//    std::cout << "Allocated " << ((float)n*n*sizeof(float))/1024/1024 << " MB" << std::endl;
//
//    // Initialize the matrices
//    auto flat_start = std::chrono::high_resolution_clock::now();
//    Flat<float> flat(n, n, data.data());
//    auto hilbert_start = std::chrono::high_resolution_clock::now();
//    HilbertCurve<float> hilbert(n, n, data.data());
//    auto zcurve_start = std::chrono::high_resolution_clock::now();
//    ZCurve<float> zcurve(n, n, data.data());
//    auto zcurvelut_start = std::chrono::high_resolution_clock::now();
//    ZCurveLUT<float> zcurvelut(n, n, data.data());
//    auto end = std::chrono::high_resolution_clock::now();
//
//    // Print the time it took to initialize
//    std::vector<std::string> names = { flat.name, hilbert.name, zcurve.name, zcurvelut.name };
//    std::vector<uint64_t> times = {
//        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(hilbert_start - flat_start).count(),
//        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(zcurve_start - hilbert_start).count(),
//        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(zcurvelut_start - zcurve_start).count(),
//        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - zcurvelut_start).count()
//    };
//    pretty_print("initialization", names, times);
//
//    // Test flat indexing
//    times = {
//        bench_flat_access(data, flat, n, n),
//        bench_flat_access(data, hilbert, n, n),
//        bench_flat_access(data, zcurve, n, n),
//        bench_flat_access(data, zcurvelut, n, n)
//    };
//    pretty_print("flat indexing", names, times);
//
//    // Test transposed indexing
//    times = {
//        bench_trans_access(data, flat, n, n),
//        bench_trans_access(data, hilbert, n, n),
//        bench_trans_access(data, zcurve, n, n),
//        bench_trans_access(data, zcurvelut, n, n)
//    };
//    pretty_print("transposed indexing", names, times);
//
//    // Test 2D sum
//    std::vector<int64_t> rs = {1, 2, 4, 8};
//    for (auto r : rs) {
//        times[0] = bench_2d_sum(flat, n, n, r);
//        times[1] = bench_2d_sum(hilbert, n, n, r);
//        times[2] = bench_2d_sum(zcurve, n, n, r);
//        times[3] = bench_2d_sum(zcurvelut, n, n, r);
//        pretty_print("2D sum (r = " + std::to_string(r) + ")", names, times);
//    }
//}