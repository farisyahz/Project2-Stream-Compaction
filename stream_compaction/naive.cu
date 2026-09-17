#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // One Hillis-Steele step, reading and writing separate buffers.
        __global__ void kernScanStep(int n, int strides, int *odata, const int *idata){
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) return;

            if (index >= strides) {
                odata[index] = idata[index] + idata[index - strides];
            } else {
                odata[index] = idata[index];
            }
        }

        __global__ void kernShiftExclusive(int n, int* odata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < n) {
                odata[index] = index == 0 ? 0 : idata[index -
                1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_out;
            int *dev_in;
            size_t bytes = n * sizeof(int);

            if (n <= 0) return;

            cudaMalloc(&dev_out, bytes);
            cudaMalloc(&dev_in, bytes);

            cudaMemcpy(dev_in, idata, bytes, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            const int threadsPerBlock = Common::blockSize() > 0 ? Common::blockSize() : 128;
            const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            for (int stride = 1; stride < n; stride *= 2){
                kernScanStep<<<blocks, threadsPerBlock>>>(n, stride, dev_out, dev_in);
                checkCUDAError("kernScanStep failed");

                int *temp = dev_out;
                dev_out = dev_in;
                dev_in = temp;
            }

            kernShiftExclusive<<<blocks, threadsPerBlock>>>(n, dev_out, dev_in);
            checkCUDAError("kernShiftExclusive failed");
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, bytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_out);
            cudaFree(dev_in);
        }
    }
}
