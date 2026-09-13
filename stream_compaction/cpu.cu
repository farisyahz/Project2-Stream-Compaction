#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        static void scanImpl(int n, int *odata, const int *idata) {
            int acc = 0;
            for (int i = 0; i < n; ++i){
                odata[i] = acc;
                acc += idata[i];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanImpl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int idx = 0;
            for (int i = 0; i < n; ++i){
                if (idata[i] != 0) {
                    odata[idx] = idata[i];
                    ++idx;
                }
            }

            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) return 0;
            std::vector<int> bools(n), indices(n);
            timer().startCpuTimer();
            for (int i = 0; i < n; ++i){
                bools[i] = idata[i] != 0;
            }
            scanImpl(n, indices.data(), bools.data());

            // scatter
            for (int i = 0; i < n; ++i){
                if (idata[i] != 0){
                    odata[indices[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return indices[n - 1] + bools[n - 1];
        }
    }
}
