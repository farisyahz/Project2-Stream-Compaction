#include "radix.h"
#include "efficient_internal.h"

namespace StreamCompaction {
    namespace Radix {
        Common::PerformanceTimer& timer() {
            static Common::PerformanceTimer value;
            return value;
        }

        __device__ int zeroBit(int value, int bit) {
            unsigned int key = static_cast<unsigned int>(value) ^ 0x80000000u;
            return ((key >> bit) & 1u) == 0;
        }

        __global__ void kernBitMask(int n, int padded, int bit, const int *input, int *mask) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < padded) mask[index] = index < n ? zeroBit(input[index], bit) : 0;
        }

        __global__ void kernPartition(int n, int bit, const int *input, const int *indices, int *output) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int totalZeros = indices[n - 1] + zeroBit(input[n - 1], bit);
            int destination = zeroBit(input[index], bit) ? indices[index] : totalZeros + index - indices[index];
            output[destination] = input[index];
        }

        void sort(int n, int *odata, const int *idata) {
            if (n <= 0) return;
            int padded = 1 << ilog2ceil(n);
            int threads = Common::blockSize() > 0 ? Common::blockSize() : 256;
            int *input;
            int *output;
            int *indices;

            cudaMalloc(&input, n * sizeof(int));
            cudaMalloc(&output, n * sizeof(int));
            cudaMalloc(&indices, padded * sizeof(int));
            cudaMemcpy(input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Radix setup failed");

            timer().startGpuTimer();
            for (int bit = 0; bit < 32; ++bit) {
                kernBitMask<<<(padded + threads - 1) / threads, threads>>>(n, padded, bit, input, indices);
                checkCUDAError("Radix bit mask failed");
                Efficient::scanDevice(padded, indices);
                kernPartition<<<(n + threads - 1) / threads, threads>>>(n, bit, input, indices, output);
                checkCUDAError("Radix partition failed");
                std::swap(input, output);
            }
            timer().endGpuTimer();
            
            cudaMemcpy(odata, input, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Radix download failed");
            cudaFree(input);
            cudaFree(output);
            cudaFree(indices);
        }
    }
}
