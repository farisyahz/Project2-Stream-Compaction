#include <cstdio>
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
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int acc = 0;
            for (int i = 0; i < n; ++i){
                odata[i] = acc;
                acc += idata[i];
            }
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
            timer().startCpuTimer();
            // Scan to odata first
            int idx = 0;
            for (int i = 0; i < n; ++i){
                odata[i] = idx;
                if (idata[i] != 0){
                    ++idx;
                }
            }

            // scatter
            for (int i = 0; i < n; ++i){
                if (idata[i] != 0){
                    odata[odata[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return idx;
        }
    }
}
