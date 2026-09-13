#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            thrust::host_vector<int> hostInput(idata, idata + n);
            thrust::device_vector<int> devInput = hostInput;
            thrust::device_vector<int> devOutput(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(devInput.begin(), devInput.end(), devOutput.begin());
            timer().endGpuTimer();

            thrust::copy(devOutput.begin(), devOutput.end(), odata);
        }
    }
}
