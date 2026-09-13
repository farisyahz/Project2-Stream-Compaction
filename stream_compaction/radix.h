#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        Common::PerformanceTimer& timer();
        void sort(int n, int *odata, const int *idata);
    }
}
