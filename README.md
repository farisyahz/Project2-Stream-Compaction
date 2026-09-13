# CUDA Stream Compaction

University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1 - Flocking

Faris Rafie Syahzani

Tested on: Windows 11, AMD Ryzen 7 8845HS, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), 16 GB RAM

[How it works](#how-it-works) · [Results](#results) · [Reproduce](#reproduce) · [Correctness](#correctness) · [Next evidence](#next-evidence)

This project builds exclusive prefix sums and uses them to remove zeros while preserving element order. The same idea lets a renderer discard finished rays so that later work focuses on rays still in flight.

```text
Input        [1, 5, 0, 1, 2, 0, 3]
Scan         [0, 1, 6, 6, 7, 9, 9]
Compaction   [1, 5, 1, 2, 3]          count = 5
```

## How it works

An **exclusive scan** writes the sum of everything before an element. For compaction, scan a mask of zeros and ones: each retained element then knows its destination.

```text
Input          [1, 5, 0, 1, 2, 0, 3]
Nonzero mask   [1, 1, 0, 1, 1, 0, 1]
Scan of mask   [0, 1, 2, 2, 3, 4, 4]
                     ↓ scatter retained values
Output         [1, 5, 1, 2, 3]
```

The count is the last scanned index plus the last mask value. Negative values are retained; only zeros are removed.

| Implementation | Approach | Total work |
|---|---|---|
| CPU scan | Serial running sum | O(n) |
| Naive GPU scan | Ping-pong buffers; offsets 1, 2, 4, …; exclusive shift | O(n log n) |
| Efficient GPU scan | Blelloch up-sweep, zero root, down-sweep | O(n) |
| Thrust scan | `thrust::exclusive_scan` on device vectors | Library-managed |
| CPU compaction | Direct filtering or map → shared scan helper → scatter | O(n) |
| GPU compaction | Common map kernel → efficient device scan → common scatter kernel | O(n) |

Efficient scan pads with zeros to the next power of two. Its grid shrinks with the active tree nodes at each level. Independent nodes operate in place, with separate launches ordering the levels. CPU scan and compaction share an untimed scan helper to avoid nested timers.

Empty inputs return without touching output. Only the first returned `count` elements of compacted output are meaningful. Values and sums must fit in `int`; arbitrary overlapping scan buffers are not supported.

## Results

**Efficient scan wins at the largest tested size, but not at every size.** At 4,194,304 integers it took 0.881 ms, versus 4.242 ms for naive and 2.057 ms for CPU: about **4.81×** and **2.33×** faster. At 1,048,576 integers, naive scan was faster than efficient scan.

![Scan times across input sizes, with four consistently colored implementations](img/scan-scaling.png)

Both axes are logarithmic. Lines show medians of 15 samples; faint bands show sorted samples 4 through 12, approximately the middle half—not confidence intervals. Lower is better. These are timed computation regions, **not end-to-end upload/compute/download latency**.

| Input integers | CPU (ms) | Naive (ms) | Efficient (ms) | Thrust (ms) |
|---:|---:|---:|---:|---:|
| 256 | 0.000300 | 0.222208 | 0.231360 | 0.060416 |
| 1,048,576 | 0.493200 | 0.392544 | 0.605536 | 0.730112 |
| 1,048,577 | 0.694900 | 0.419488 | 0.640192 | 0.743424 |
| 4,194,304 | 2.056500 | 4.242340 | 0.881312 | 0.920736 |

These describe one machine/session, not universal rankings. All samples are retained in [analysis/results.json](analysis/results.json).

### Tune before comparing

![Block-size sweep at one million integers, with selected sizes highlighted](img/block-sweep.png)

We swept 64, 128, 256, and 512 threads per block. The lowest median at 1,048,576 elements selected **512 for naive** and **256 for efficient**. Those configurations stay fixed across the scaling plot and are the defaults in the source. Thrust manages its own launches; the block field in its result records is ignored. Compaction inherits the efficient setting, but has not undergone separate performance tuning.

This is rough tuning at one reference size. Naive's 128- and 512-thread results are close; repeat before interpreting that small difference. The best setting can change with input size and hardware.

### Understanding the curves

**Small arrays favor the CPU.** A serial loop avoids GPU launch and scheduling costs. CUDA event intervals include gaps between launches, so our measurements are not sums of pure kernel durations. The fixed-cost explanation is plausible, but its contribution has not been measured separately.

**Less work does not guarantee less time.** For `n = 2^k`, naive launches `k` scan kernels plus one shift, while efficient launches `2k` kernels plus a root reset. At one million elements that is 21 versus 40 kernel launches. Efficient repeatedly launches tiny grids near the root. This can explain why naive wins at medium sizes despite doing more additions. [GPU Gems Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) derives this work-complexity distinction.

**At the largest size, fewer full-array passes pay off.** Naive rereads and rewrites nearly the entire array at each level. Efficient processes progressively fewer nodes. The crossover is consistent with that memory-traffic difference. Bandwidth has not been measured; we cannot claim that either kernel saturates it. Efficient's increasingly strided accesses also mean that fewer additions need not imply equally efficient memory transactions.

**Doubling padding does not double runtime here.** Going from 1,048,576 to 1,048,577 doubles efficient scan's padded array, but its median rises only about 5.7%. CPU timing also shifts substantially between these adjacent sizes, indicating that cache/allocation state and noise complicate interpretation. This probe is reported separately rather than squeezed into the scaling graph.

**Thrust's step needs profiler evidence.** Its measured cost rises sharply between 65,536 and 262,144 elements. We have not established why: internal dispatch, temporary allocations, synchronization, and Windows scheduling are candidates. Internal activity inside the scan call remains timed even though explicit vector construction and the final copy are excluded. Thrust's large-input result is close to efficient scan; this run does not establish a general advantage over Thrust.

### Measurement conditions

| Item | Recorded configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB reported |
| CPU | AMD Ryzen 7 8845HS |
| Platform | Windows, Visual Studio 2022, MSVC 19.41 |
| CUDA / driver | 13.3 / 616.56 |
| Build | Release, C++17, no debugger, native GPU architecture |
| Sampling | 3 warm-ups + 15 samples per configuration |
| Input | Seed 565, integers 0–3; same input per size across methods |
| Timer | Provided chrono timer for CPU; CUDA events for GPU |

Explicit initial allocations/uploads and final downloads/frees are excluded. Validation and synchronization checks run outside timers. Power mode, AC/battery state, and background activity were not controlled. Configurations run in fixed order; cache state, clocks, thermal effects, and driver scheduling may bias results. Tiny CPU durations approach timer resolution. See the [analysis plan](analysis/PLAN.md) for a more controlled repeat.

## Reproduce

From the repository root, with CUDA and Visual Studio C++ tools installed:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\bin\Release\analysis.exe
.\build\bin\Release\analysis.exe analysis/results.json
python scripts/plot_results.py
```

Without a path, `analysis.exe` runs correctness checks only. With a path it also benchmarks and writes raw samples. The plotting script requires Matplotlib (3.9.2 in this run), applies the documented selection rule, and regenerates both figures. Update this table and discussion if you replace the dataset.

Run the original starter program with `.\build\bin\Release\cis5650_stream_compaction_test.exe`.

**CMake modifications:** the root configuration adds the `analysis` executable. The library passes `/Zc:preprocessor` through NVCC only under MSVC, as required by this CUDA/Thrust installation. During automated compilation, duplicate `PATH`/`Path` environment entries caused an MSBuild error; normalizing the child-process environment allowed Release to build.

## Correctness

Independent `std::exclusive_scan` and `std::copy_if` references check values, counts, and an output sentinel. Every timed scan output is checked as well, across all four block-size candidates.

```text
PASS: 336 independent scan/compaction checks
Sizes: 0, 1, 2, 3, 127, 128, 129, 253, 256, 257, 10000, 1000000
Patterns: zeros, ones, mixed signed values, trailing zero
Measured n=256
Measured n=1024
Measured n=4096
Measured n=16384
Measured n=65536
Measured n=262144
Measured n=1048576
Measured n=1048577
Measured n=4194304
```

The large-input sweep exposed inactive-thread index overflow in the tree passes. They now return before calculating an inactive node's index; the entire sweep then passed. No memory-sanitizer run is claimed.


<details>
<summary>Starter test program output (Release)</summary>

These single-call timings are correctness-run output, not the repeated samples used in the graphs.

```text
****************
** SCAN TESTS **
****************
    [  21  13   3   2  37  29   1   5  26  16  43  37  26 ...  32   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   0  21  34  37  39  76 105 106 111 137 153 196 233 ... 5931 5963 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0002ms    (std::chrono Measured)
    [   0  21  34  37  39  76 105 106 111 137 153 196 233 ... 5861 5893 ]
    passed 
==== naive scan, power-of-two ====
   elapsed time: 0.321536ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 0.666624ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.494592ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.231424ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.20992ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.073728ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   1   2   0   1   0   0   1   3   3   3   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   3   1   2   1   1   3   3   3   1   3   3   2   2 ...   1   1 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   3   1   2   1   1   3   3   3   1   3   3   2   2 ...   3   3 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 0.0012ms    (std::chrono Measured)
    [   3   1   2   1   1   3   3   3   1   3   3   2   2 ...   1   1 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 0.407552ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.41472ms    (CUDA Measured)
Press any key to continue . . . 
    passed
```

</details>

## Next evidence

**The assignment's Thrust timeline question remains pending.** Add an Nsight Systems capture of a warmed Thrust scan with CUDA API calls, GPU kernels, neighboring allocations/copies, and a readable time axis. Captures around 65,536 and 262,144 elements would be especially informative. A second view near efficient scan's root would test the launch-overhead explanation.

For stronger bandwidth/occupancy claims, capture Nsight Compute's Launch Statistics, Memory Workload Analysis, and Speed Of Light views for an early/late tree pass and a naive pass. Profile separately from benchmarking. The [analysis plan](analysis/PLAN.md) details the evidence to collect; see the official [Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/) and [Compute guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Feature scope

Required scans, compaction, and Thrust are implemented. Efficient scan already follows Part 5's shrinking-grid strategy, but no isolated baseline speedup is claimed. Optional radix sort and shared-memory scan are not implemented or claimed. Final submission and the pull request remain manual.

