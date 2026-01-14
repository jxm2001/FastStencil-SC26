#include "defines.h"
#include "logger.hpp"
#include "SpatialTiling/params.cuh"
#define Offset_sm(i,j) ((j) + sm_x_tile * (i))

using namespace SpatialTilingParams;

template<KernelType KERNEL_T, typename VALUE_T>
__global__ void SpatialTiling_Kernel(VALUE_T* __restrict gm_input, VALUE_T* __restrict gm_output, int NY, int NX){
	extern __shared__ char sm[];
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	constexpr int threads_y_num = DefaultParams::threads_y_num;
	constexpr int threads_x_num = DefaultParams::threads_x_num;
	constexpr int CTX_SZ = threads_y_num * threads_x_num;
	constexpr int gm_y_vaild_tile = threads_y_num;
	constexpr int gm_x_vaild_tile = threads_x_num;
	constexpr int gm_y_halo_tile = gm_y_vaild_tile + RAD * 2;
	constexpr int gm_x_halo_tile = gm_x_vaild_tile + RAD * 2;
	constexpr int sm_y_tile = gm_y_halo_tile;
	constexpr int sm_x_tile = gm_x_halo_tile;

	const int thread_idx = threads_x_num * threadIdx.y + threadIdx.x;
	const int gm_y_vaild_begin = blockIdx.y * gm_y_vaild_tile;
	const int gm_y_vaild_end = min(gm_y_vaild_begin + gm_y_vaild_tile, NY);
	const int gm_y_halo_begin = gm_y_vaild_begin - RAD;
	const int gm_y_halo_end = gm_y_vaild_end + RAD;
	const int gm_x_vaild_begin = blockIdx.x * gm_x_vaild_tile;
	const int gm_x_vaild_end = min(gm_x_vaild_begin + gm_x_vaild_tile, NX);
	const int gm_x_halo_begin = gm_x_vaild_begin - RAD;
	const int gm_x_halo_end = gm_x_vaild_end + RAD;

	VALUE_T *sm_data = ((VALUE_T*)sm);
	VALUE_T reg_input[RAD * 2 + 1][RAD * 2 + 1], reg_output = 0;

	for(int i = thread_idx; i < sm_y_tile * sm_x_tile; i += CTX_SZ){
		int data_y = i / sm_x_tile;
		int sm_y = data_y;
		int gm_y = min(max(gm_y_halo_begin + data_y, 0), NY - 1);
		int data_x = i % sm_x_tile;
		int sm_x = data_x;
		int gm_x = min(max(gm_x_halo_begin + data_x, 0), NX - 1);
		sm_data[Offset_sm(sm_y, sm_x)] = gm_input[Offset_gm(gm_y, gm_x)];
	}
	__syncthreads();
	#pragma unroll
	for(int data_y = 0; data_y < RAD * 2 + 1; data_y++){
		int sm_y = data_y + threadIdx.y;
		#pragma unroll
		for(int data_x = 0; data_x < RAD * 2 + 1; data_x++){
			int sm_x = data_x + threadIdx.x;
			reg_input[data_y][data_x] = sm_data[Offset_sm(sm_y, sm_x)];
		}
	}
	utils::kernelCompute<KERNEL_T, VALUE_T>(&reg_output, reg_input);
	{
		int gm_y = gm_y_vaild_begin + threadIdx.y;
		int gm_x = gm_x_vaild_begin + threadIdx.x;
		if(gm_y < gm_y_vaild_end && gm_x < gm_x_vaild_end){
			gm_output[Offset_gm(gm_y, gm_x)] = reg_output;
		}
	}
}

template<KernelType KERNEL_T, typename VALUE_T>
void SpatialTiling(VALUE_T *init_data, VALUE_T *output_data, int NY, int NX, int T, bool enable_warmup){
	int device, sm_count, numBlocksPerSm_current;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	constexpr int threads_y_num = DefaultParams::threads_y_num;
	constexpr int threads_x_num = DefaultParams::threads_x_num;
	constexpr int CTX_SZ = threads_y_num * threads_x_num;
	constexpr int gm_y_vaild_tile = threads_y_num;
	constexpr int gm_x_vaild_tile = threads_x_num;
	constexpr int gm_y_halo_tile = gm_y_vaild_tile + RAD * 2;
	constexpr int gm_x_halo_tile = gm_x_vaild_tile + RAD * 2;

	const int grid_dim_y = (NY + gm_y_vaild_tile - 1) / gm_y_vaild_tile;
	const int grid_dim_x = (NX + gm_x_vaild_tile - 1) / gm_x_vaild_tile;
	const size_t sm_size = sizeof(VALUE_T) * (gm_y_halo_tile * gm_x_halo_tile);

	if(sm_size > props.sharedMemPerBlockOptin){
		LOG_ERROR("requested shared memory size (%lu bytes) exceeds device limit (%lu bytes)", sm_size, props.sharedMemPerBlockOptin);
		exit(1);
	}

	auto kernel_func = SpatialTiling_Kernel<KERNEL_T, VALUE_T>;
	cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, props.sharedMemPerBlockOptin);
	cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_current, kernel_func, CTX_SZ, sm_size);

	const int total_blocks   = grid_dim_x * grid_dim_y;
	const int blocks_per_sm  = total_blocks / sm_count;
	const double blocks_per_sm_real  = 1.0 * total_blocks / sm_count;
	const int threads_per_sm = CTX_SZ * numBlocksPerSm_current;
	if (total_blocks % sm_count != 0) {
		LOG_WARNING( "grid (%d blocks) is not evenly divisible by %d SMs, load imbalance may occur", total_blocks, sm_count);
	}
	if (blocks_per_sm < numBlocksPerSm_current) {
		LOG_WARNING( "only %.2lf blocks/SM < required %d blocks/SM for full occupancy, SMs may be underutilized", blocks_per_sm_real, numBlocksPerSm_current);
	}
	if (!(threads_per_sm % 128 == 0 && threads_per_sm >= 256)) {
		LOG_WARNING( "active threads per SM (%d) not 128-aligned or <256, SMs may be underutilized", threads_per_sm);
	}

	VALUE_T *gm_input, *gm_output;
	size_t data_memsize = sizeof(VALUE_T)*NY*NX;
	cudaMalloc(&gm_input, data_memsize);
	cudaMalloc(&gm_output, data_memsize);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	LOG_INFO("grid_dim_x: %d, grid_dim_y: %d, CTX_SZ: %d, sm_size: %lu", grid_dim_x, grid_dim_y, CTX_SZ, sm_size);

	dim3 gridDim(grid_dim_x, grid_dim_y), blockDim(threads_x_num, threads_y_num);
	if(enable_warmup){
		cudaEventRecord(start);
		kernel_func<<<gridDim, blockDim, sm_size>>>(gm_input, gm_output, NY, NX);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float single_run_time = 0;
		cudaEventElapsedTime(&single_run_time, start, stop);

		const float target_warmup_time = 5000.0f;
		int estimated_iterations = 0;

		if(single_run_time > 0) {
			estimated_iterations = static_cast<int>(target_warmup_time / single_run_time);
		}

		int warmup_iterations = std::max(50, estimated_iterations);

		LOG_INFO("Warmup iterations: %d (estimated for ~5s)", warmup_iterations);

		for(int warmup_iter = 0; warmup_iter < warmup_iterations; warmup_iter++){
			kernel_func<<<gridDim, blockDim, sm_size>>>(gm_input, gm_output, NY, NX);
			cudaDeviceSynchronize();
		}
	}

	cudaMemcpy(gm_input, init_data, data_memsize, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for(int t = 0; t < T; t++){
		if(t != 0)
			std::swap(gm_input, gm_output);
		kernel_func<<<gridDim, blockDim, sm_size>>>(gm_input, gm_output, NY, NX);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
	double gstencils = 1.0 * NY * NX * T / elapsedTime / 1e6;
	double tflops = 1e-3 * gstencils * KernelInfo<KERNEL_T>::FP;
	LOG_CRITICAL("SpatialTiling: %lf GStencil/s, %lf TFLOPS", gstencils, tflops);

	cudaMemcpy(output_data, gm_output, data_memsize, cudaMemcpyDeviceToHost);

	cudaFree(gm_input);
	cudaFree(gm_output);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
template void SpatialTiling<KernelType::j2d5pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void SpatialTiling<KernelType::j2d9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void SpatialTiling<KernelType::j2ds9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void SpatialTiling<KernelType::j2d25pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void SpatialTiling<KernelType::j2d13pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void SpatialTiling<KernelType::j2d49pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
