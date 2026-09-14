#include <stream_compaction/shared.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/naive.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace StreamCompaction;
    if (argc == 4 && std::string(argv[1]) == "--profile") {
        int size = std::stoi(argv[3]);
        if (size <= 0 || size > 16777216) return 1;
        std::vector<int> input(size, 1), output(size);
        std::string method = argv[2];
        auto scan = method == "thrust" ? Thrust::scan : method == "efficient" ? Efficient::scan : method == "shared" ? Shared::scan : method == "shared-unpadded" ? Shared::scanUnpadded : method == "naive" ? Naive::scan : nullptr;
        if (!scan) return 1;
        for (int warmup = 0; warmup < 3; ++warmup) scan(size, output.data(), input.data());
        cudaProfilerStart();
        scan(size, output.data(), input.data());
        cudaProfilerStop();
        return cudaDeviceSynchronize() == cudaSuccess && output.back() == size - 1 ? 0 : 1;
    }
    std::mt19937 random(565);
    int checks = 0;
    for (int block : {64, 128, 256, 512}) {
        Common::blockSize() = block;
        for (int size : {0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 10000, 1000000}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<int> input(size), expected(size), output(size + 1, -999);
                for (int& value : input) value = pattern == 0 ? 0 : pattern == 1 ? 1 : static_cast<int>(random() % 11) - 5;
                std::exclusive_scan(input.begin(), input.end(), expected.begin(), 0);
                for (auto scan : {Shared::scan, Shared::scanNaive, Shared::scanUnpadded, Efficient::scanUnoptimized}) {
                    std::fill(output.begin(), output.end(), -999);
                    scan(size, output.data(), input.data());
                    if (cudaDeviceSynchronize() != cudaSuccess || !std::equal(expected.begin(), expected.end(), output.begin()) || output[size] != -999) {
                        std::cerr << "Scan failed: size=" << size << " block=" << block << '\n';
                        return 1;
                    }
                    ++checks;
                }
            }
        }
    }
    Common::blockSize() = 0;
    for (int size : {0, 1, 2, 3, 127, 128, 129, 1025, 10000, 1000000}) {
        for (int pattern = 0; pattern < 5; ++pattern) {
            std::vector<int> input(size), output(size + 1, -999);
            for (int& value : input) value = pattern == 0 ? 0 : static_cast<int>(random() % 20001) - 10000;
            if (size > 1) { input.front() = INT_MIN; input.back() = INT_MAX; }
            if (pattern == 2) std::sort(input.begin(), input.end());
            if (pattern == 3) std::sort(input.rbegin(), input.rend());
            if (pattern == 4) std::fill(input.begin(), input.end(), -7);
            auto expected = input;
            std::stable_sort(expected.begin(), expected.end());
            Radix::sort(size, output.data(), input.data());
            if (cudaDeviceSynchronize() != cudaSuccess || !std::equal(expected.begin(), expected.end(), output.begin()) || output[size] != -999) {
                std::cerr << "Radix failed: size=" << size << '\n';
                return 1;
            }
            ++checks;
        }
    }
    int input[] = {3, -1, 0, 3, -7, 2};
    int output[6];
    Radix::sort(6, output, input);
    std::cout << "Radix example:";
    for (int value : output) std::cout << ' ' << value;
    std::cout << "\nPASS: " << checks << " extra-credit checks\n";
    return 0;
}
