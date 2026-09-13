#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int stride, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int right = (index + 1) * stride * 2 - 1;

            if (right < n) {
                data[right] += data[right - stride];
            }
        }

        __global__ void kernDownSweep(int n, int stride, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int right = (index + 1) * stride * 2 - 1;

            if (right < n) {
                int leftValue = data[right - stride];
                data[right - stride] = data[right];
                data[right] += leftValue;
            }
        }

        void scanDevice(int n, int *data) {
            const int threadsPerBlock = 128;

            for (int stride = 1; stride < n; stride *= 2) {
                int activeThreads = n / (stride * 2);
                int blocks = (activeThreads + threadsPerBlock - 1) / threadsPerBlock;
                kernUpSweep<<<blocks, threadsPerBlock>>>(n, stride, data);
                checkCUDAError("kernUpSweep failed");
            }

            cudaMemset(data + n - 1, 0, sizeof(int));

            for (int stride = n / 2; stride >= 1; stride /= 2) {
                int activeThreads = n / (stride * 2);
                int blocks = (activeThreads + threadsPerBlock - 1) / threadsPerBlock;
                kernDownSweep<<<blocks, threadsPerBlock>>>(n, stride, data);
                checkCUDAError("kernDownSweep failed");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);
            size_t inputBytes = n * sizeof(int);
            size_t paddedBytes = paddedN * sizeof(int);
            int *devData;

            cudaMalloc(&devData, paddedBytes);
            cudaMemset(devData, 0, paddedBytes);
            cudaMemcpy(devData, idata, inputBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanDevice(paddedN, devData);
            timer().endGpuTimer();

            cudaMemcpy(odata, devData, inputBytes, cudaMemcpyDeviceToHost);
            cudaFree(devData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);
            size_t inputBytes = n * sizeof(int);
            size_t paddedBytes = paddedN * sizeof(int);
            int *devInput;
            int *devOutput;
            int *devBools;
            int *devIndices;

            cudaMalloc(&devInput, inputBytes);
            cudaMalloc(&devOutput, inputBytes);
            cudaMalloc(&devBools, paddedBytes);
            cudaMalloc(&devIndices, paddedBytes);

            cudaMemcpy(devInput, idata, inputBytes, cudaMemcpyHostToDevice);
            cudaMemset(devBools, 0, paddedBytes);

            timer().startGpuTimer();
            const int threadsPerBlock = 128;
            const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            Common::kernMapToBoolean<<<blocks, threadsPerBlock>>>(n, devBools, devInput);
            checkCUDAError("kernMapToBoolean failed");
            cudaMemcpy(devIndices, devBools, paddedBytes, cudaMemcpyDeviceToDevice);

            scanDevice(paddedN, devIndices);
            Common::kernScatter<<<blocks, threadsPerBlock>>>(
                n, devOutput, devInput, devBools, devIndices);
            checkCUDAError("kernScatter failed");
            timer().endGpuTimer();

            int lastIndex;
            int lastBool;
            cudaMemcpy(&lastIndex, devIndices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, devBools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, devOutput, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(devInput);
            cudaFree(devOutput);
            cudaFree(devBools);
            cudaFree(devIndices);

            return count;
        }
    }
}
