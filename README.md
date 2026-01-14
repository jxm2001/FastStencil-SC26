# FastStencil Artifact (SC'26)

This repository provides the artifact for reproducing the experimental results of the FastStencil paper (SC'26). It allows users to build the implementation, run performance experiments, and reproduce the figures and tables reported in the paper.

## Requirements

### Hardware
- NVIDIA A100 80GB SXM
- NVIDIA H100 SXM5  
- Single GPU only

### Software

- **CUDA Toolkit 12.9.1 (required for performance reproduction)**

  Using CUDA Toolkit 12.9.1 is strongly required to reproduce the reported performance. Different CUDA versions may employ different register allocation strategies. Since this implementation fully utilizes available registers, changes in register allocation may lead to register spilling, resulting in performance degradation or even runtime errors.
- CMake ≥ 3.25  
- Python 3 (with standard scientific libraries)

## Reproducing Results

### A100

```bash
git clone https://github.com/jxm2001/FastStencil-SC26
cd FastStencil-SC26
git submodule update --init
# Compile all binaries
bash script/build.sh
# Run benchmarks and collect performance data
python script/perf.py perf A100
# Collect Nsight Compute profiling results
python script/perf.py ncu A100
# Generate performance plots (TFLOPS figures)
python plot-tflops.py A100
# Convert profiling results to CSV format
python ncu2csv.py A100
# Generate tables reported in the paper
python analyze_data.py A100
````

### H100

```bash
git clone https://github.com/jxm2001/FastStencil-SC26
cd FastStencil-SC26
git submodule update --init
# Compile all binaries
bash script/build.sh
# Run benchmarks and collect performance data
python script/perf.py perf H100
# Collect Nsight Compute profiling results
python script/perf.py ncu H100
# Generate performance plots (TFLOPS figures)
python plot-tflops.py H100
# Convert profiling results to CSV format
python ncu2csv.py H100
# Generate tables reported in the paper
python analyze_data.py H100
```

## Output

For each GPU platform (`A100` or `H100`):

* Visualization results:

  ```
  output/<platform>/images
  ```

* Processed performance data:

  ```
  output/<platform>/analysis.csv
  ```

These outputs correspond directly to the figures and tables presented in the paper. 

## Notes

* Each configuration is executed multiple times with warm-up runs
* The **peak performance** across repeated runs is reported

## Expected Runtime

* Setup: ~5 minutes
* Execution: ~120 minutes (A100 or H100)
* Analysis: ~1 minute
