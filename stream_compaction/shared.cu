#include "shared.h"
#include <vector>

namespace StreamCompaction {
    namespace Shared {
        Common::PerformanceTimer& timer() {
            static Common::PerformanceTimer value;
            return value;
        }

        __global__ void kernNaiveBlocks(int n, const int *input, int *output, int *sums) {
            extern __shared__ int storage[];
            int local = threadIdx.x;
            int index = blockIdx.x * blockDim.x + local;
            int *current = storage;
            int *next = storage + blockDim.x;
            current[local] = index < n ? input[index] : 0;
            __syncthreads();
            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                next[local] = current[local] + (local >= stride ? current[local - stride] : 0);
                __syncthreads();
                int *temporary = current;
                current = next;
                next = temporary;
            }
            if (index < n) output[index] = local == 0 ? 0 : current[local - 1];
            if (local == blockDim.x - 1 && sums) sums[blockIdx.x] = current[local];
        }

        template<bool Padded>
        __device__ int address(int index) {
            return index + (Padded ? index / 32 : 0);
        }

        template<bool Padded>
        __global__ void kernTreeBlocks(int n, const int *input, int *output, int *sums) {
            extern __shared__ int storage[];
            int local = threadIdx.x;
            int width = blockDim.x * 2;
            int base = blockIdx.x * width;
            int second = local + blockDim.x;
            storage[address<Padded>(local)] = base + local < n ? input[base + local] : 0;
            storage[address<Padded>(second)] = base + second < n ? input[base + second] : 0;
            __syncthreads();
            for (int stride = 1; stride < width; stride *= 2) {
                int right = (local + 1) * stride * 2 - 1;
                if (right < width) storage[address<Padded>(right)] += storage[address<Padded>(right - stride)];
                __syncthreads();
            }
            if (local == 0) {
                if (sums) sums[blockIdx.x] = storage[address<Padded>(width - 1)];
                storage[address<Padded>(width - 1)] = 0;
            }
            __syncthreads();
            for (int stride = width / 2; stride > 0; stride /= 2) {
                int right = (local + 1) * stride * 2 - 1;
                if (right < width) {
                    int left = address<Padded>(right - stride);
                    right = address<Padded>(right);
                    int temporary = storage[left];
                    storage[left] = storage[right];
                    storage[right] += temporary;
                }
                __syncthreads();
            }
            if (base + local < n) output[base + local] = storage[address<Padded>(local)];
            if (base + second < n) output[base + second] = storage[address<Padded>(second)];
        }

        __global__ void kernAddOffsets(int n, int width, int *output, const int *offsets) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < n) output[index] += offsets[index / width];
        }

        static void scanLevel(int n, const int *input, int *output, int threads, int mode,
                              const std::vector<int*>& sums, const std::vector<int*>& offsets, int level) {
            int width = mode == 0 ? threads : threads * 2;
            int blocks = (n + width - 1) / width;
            int *totals = blocks > 1 ? sums[level] : nullptr;
            if (mode == 0) {
                kernNaiveBlocks<<<blocks, threads, threads * 2 * sizeof(int)>>>(n, input, output, totals);
            } else if (mode == 1) {
                kernTreeBlocks<false><<<blocks, threads, width * sizeof(int)>>>(n, input, output, totals);
            } else {
                kernTreeBlocks<true><<<blocks, threads, (width + width / 32) * sizeof(int)>>>(n, input, output, totals);
            }
            checkCUDAError("Shared block scan failed");
            if (blocks > 1) {
                scanLevel(blocks, sums[level], offsets[level], threads, mode, sums, offsets, level + 1);
                kernAddOffsets<<<(n + threads - 1) / threads, threads>>>(n, width, output, offsets[level]);
                checkCUDAError("Shared offset addition failed");
            }
        }

        static void scanHost(int n, int *odata, const int *idata, int mode) {
            if (n <= 0) return;
            int threads = Common::blockSize() > 0 ? Common::blockSize() : 128;
            if (threads < 32 || threads > 1024 || (threads & (threads - 1))) {
                throw std::invalid_argument("Shared scan requires a power-of-two block size between 32 and 1024");
            }
            int width = mode == 0 ? threads : threads * 2;
            int *input;
            int *output;
            cudaMalloc(&input, n * sizeof(int));
            cudaMalloc(&output, n * sizeof(int));
            std::vector<int*> sums, offsets;
            for (int count = (n + width - 1) / width; count > 1; count = (count + width - 1) / width) {
                int *total;
                int *offset;
                cudaMalloc(&total, count * sizeof(int));
                cudaMalloc(&offset, count * sizeof(int));
                sums.push_back(total);
                offsets.push_back(offset);
            }
            cudaMemcpy(input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Shared scan setup failed");
            timer().startGpuTimer();
            scanLevel(n, input, output, threads, mode, sums, offsets, 0);
            timer().endGpuTimer();
            cudaMemcpy(odata, output, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Shared scan download failed");
            cudaFree(input);
            cudaFree(output);
            for (int *buffer : sums) cudaFree(buffer);
            for (int *buffer : offsets) cudaFree(buffer);
        }

        void scan(int n, int *odata, const int *idata) {
            scanHost(n, odata, idata, 2);
        }

        void scanNaive(int n, int *odata, const int *idata) {
            scanHost(n, odata, idata, 0);
        }

        void scanUnpadded(int n, int *odata, const int *idata) {
            scanHost(n, odata, idata, 1);
        }
    }
}
