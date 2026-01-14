#include "defines.h"

int utils::getCudaArch_host() {
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	int CUDA_ARCH = props.major * 100 + props.minor * 10;

	return CUDA_ARCH;
}
