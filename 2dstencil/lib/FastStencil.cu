#include "defines.h"
#include "logger.hpp"
#include "FastStencil/params.cuh"
#include "FastStencil/kernels.cuh"
#include "FastStencil/memory-helper.cuh"

using namespace FastStencilParams;

template<KernelType KERNEL_T, typename VALUE_T>
__global__ void FastStencil_Kernel(VALUE_T* __restrict gm_input, VALUE_T* __restrict gm_output, int NY, int NX){
	extern __shared__ char sm[];
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	constexpr int UNROLL_COUNT = Params<KERNEL_T, VALUE_T>::UNROLL_COUNT;
	constexpr int core_t_tile = Params<KERNEL_T, VALUE_T>::core_t_tile;
	constexpr int core_y_tile_stream = Params<KERNEL_T, VALUE_T>::core_y_tile_stream;
	constexpr int reg_x_tile = Params<KERNEL_T, VALUE_T>::reg_x_tile;
	constexpr int warps_num = Params<KERNEL_T, VALUE_T>::warps_num;
	constexpr int CTX_SZ = warps_num * 32;
	constexpr int BT = warps_num * core_t_tile;
	constexpr int sm_y_tile = core_y_tile_stream * (warps_num + 2);
	constexpr int sm_y_tile_mod = utils::align_to_pow2(sm_y_tile);
	constexpr int gm_x_halo_tile = reg_x_tile * 32;
	constexpr int gm_x_vaild_tile = gm_x_halo_tile - RAD * BT * 2;

	const int warp_idx = threadIdx.x / 32;
	const int gm_y_vaild_tile = (NY + gridDim.y - 1) / gridDim.y;
	const int gm_y_vaild_begin = blockIdx.y * gm_y_vaild_tile;
	const int gm_y_vaild_end = min(gm_y_vaild_begin + gm_y_vaild_tile, NY);
	const int gm_y_halo_begin = gm_y_vaild_begin - RAD * BT * 2;
	const int gm_y_output_offset = (RAD * core_t_tile + core_y_tile_stream) * (warps_num - 1) + RAD * (core_t_tile - 1) + core_y_tile_stream; 
	const int gm_y_halo_end = gm_y_vaild_end + gm_y_output_offset;
	const int gm_x_vaild_begin = blockIdx.x * gm_x_vaild_tile;
	const int gm_x_vaild_end = min(gm_x_vaild_begin + gm_x_vaild_tile, NX);
	const int gm_x_halo_begin = gm_x_vaild_begin - RAD * BT;
	const int gm_x_halo_end = gm_x_vaild_end + RAD * BT;

	VALUE_T *sm_data = ((VALUE_T*)sm);
	constexpr int sm_x_tile = gm_x_halo_tile;

	constexpr int reg_y_tile_stream = core_y_tile_stream + RAD * 2;
	constexpr int reg_t_tile = core_t_tile;
	VALUE_T reg_input[reg_t_tile][reg_y_tile_stream][reg_x_tile];
	VALUE_T reg_output[reg_t_tile][core_y_tile_stream][reg_x_tile];
	int sm_y_index_for_reg, sm_y_index_for_gm_input, sm_y_index_for_gm_output;

	sm_y_index_for_reg = sm_y_tile_mod - warp_idx * core_y_tile_stream;
	sm_y_index_for_gm_input = 0;
	sm_y_index_for_gm_output = sm_y_tile_mod - warps_num * core_y_tile_stream;

	gm2sm<VALUE_T, CTX_SZ, RAD * 2, sm_y_tile_mod, sm_x_tile, sm_x_tile>
		(sm_data, gm_input, sm_y_index_for_gm_input, gm_y_halo_begin, gm_x_halo_begin, NY, NX);
	__pipeline_wait_prior(0);
	__syncthreads();
	if(warp_idx == 0){
		sm2reg<VALUE_T, RAD * 2, 0, sm_y_tile_mod, reg_x_tile, sm_x_tile>(reg_input[0], sm_data, sm_y_index_for_reg);
	}
	__syncthreads();
	gm2sm<VALUE_T, CTX_SZ, core_y_tile_stream, sm_y_tile_mod, sm_x_tile, sm_x_tile>
		(sm_data, gm_input, sm_y_index_for_gm_input, gm_y_halo_begin + RAD * 2, gm_x_halo_begin, NY, NX);
	__syncwarp();

	sm_y_index_for_gm_input += core_y_tile_stream;

	if constexpr (UNROLL_COUNT != 0){
		#pragma unroll UNROLL_COUNT
		for(int gm_y = gm_y_halo_begin + RAD; gm_y < gm_y_halo_end; gm_y += core_y_tile_stream){
			#pragma unroll
			for(int t = 0; t < reg_t_tile; t++)
				kernelCompute_pre<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[t], reg_input[t]);
			__pipeline_wait_prior(0);
			__syncthreads();
			gm2sm<VALUE_T, CTX_SZ, core_y_tile_stream, sm_y_tile_mod, sm_x_tile, sm_x_tile>
				(sm_data, gm_input, sm_y_index_for_gm_input, gm_y + core_y_tile_stream + RAD, gm_x_halo_begin, NY, NX);
			__syncwarp();
			sm2reg<VALUE_T, core_y_tile_stream, RAD * 2, sm_y_tile_mod, reg_x_tile, sm_x_tile>(reg_input[0], sm_data, sm_y_index_for_reg);
			#pragma unroll
			for(int t = 0; t < reg_t_tile - 1; t++){
				kernelCompute_post<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[t], reg_input[t]);
				reg2reg<VALUE_T, core_y_tile_stream, RAD * 2, 0, reg_x_tile>(reg_input[t + 1], reg_output[t]);
				reg2reg<VALUE_T, RAD * 2, 0, core_y_tile_stream, reg_x_tile>(reg_input[t], reg_input[t]);
			}
			kernelCompute_post<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[reg_t_tile - 1], reg_input[reg_t_tile - 1]);
			reg2sm<VALUE_T, core_y_tile_stream, 0, sm_y_tile_mod, reg_x_tile, sm_x_tile>(sm_data, reg_output[reg_t_tile - 1], sm_y_index_for_reg);
			reg2reg<VALUE_T, RAD * 2, 0, core_y_tile_stream, reg_x_tile>(reg_input[reg_t_tile - 1], reg_input[reg_t_tile - 1]);
			sm2gm<VALUE_T, CTX_SZ, core_y_tile_stream, sm_y_tile_mod, sm_x_tile, sm_x_tile>(gm_output, sm_data, sm_y_index_for_gm_output,
																				gm_y - gm_y_output_offset, gm_x_halo_begin,
																				gm_y_vaild_begin, gm_y_vaild_end, gm_x_vaild_begin, gm_x_vaild_end, NY, NX);
			__syncwarp();
			sm_y_index_for_reg += core_y_tile_stream;
			sm_y_index_for_gm_input += core_y_tile_stream;
			sm_y_index_for_gm_output += core_y_tile_stream;
		}
	}
	else{
		for(int gm_y = gm_y_halo_begin + RAD; gm_y < gm_y_halo_end; gm_y += core_y_tile_stream){
			#pragma unroll
			for(int t = 0; t < reg_t_tile; t++)
				kernelCompute_pre<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[t], reg_input[t]);
			__pipeline_wait_prior(0);
			__syncthreads();
			gm2sm<VALUE_T, CTX_SZ, core_y_tile_stream, sm_y_tile_mod, sm_x_tile, sm_x_tile>
				(sm_data, gm_input, sm_y_index_for_gm_input, gm_y + core_y_tile_stream + RAD, gm_x_halo_begin, NY, NX);
			__syncwarp();
			sm2reg<VALUE_T, core_y_tile_stream, RAD * 2, sm_y_tile_mod, reg_x_tile, sm_x_tile>(reg_input[0], sm_data, sm_y_index_for_reg);
			#pragma unroll
			for(int t = 0; t < reg_t_tile - 1; t++){
				kernelCompute_post<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[t], reg_input[t]);
				reg2reg<VALUE_T, core_y_tile_stream, RAD * 2, 0, reg_x_tile>(reg_input[t + 1], reg_output[t]);
				reg2reg<VALUE_T, RAD * 2, 0, core_y_tile_stream, reg_x_tile>(reg_input[t], reg_input[t]);
			}
			kernelCompute_post<KERNEL_T, VALUE_T, core_y_tile_stream, reg_x_tile>(reg_output[reg_t_tile - 1], reg_input[reg_t_tile - 1]);
			reg2sm<VALUE_T, core_y_tile_stream, 0, sm_y_tile_mod, reg_x_tile, sm_x_tile>(sm_data, reg_output[reg_t_tile - 1], sm_y_index_for_reg);
			reg2reg<VALUE_T, RAD * 2, 0, core_y_tile_stream, reg_x_tile>(reg_input[reg_t_tile - 1], reg_input[reg_t_tile - 1]);
			sm2gm<VALUE_T, CTX_SZ, core_y_tile_stream, sm_y_tile_mod, sm_x_tile, sm_x_tile>(gm_output, sm_data, sm_y_index_for_gm_output,
																				gm_y - gm_y_output_offset, gm_x_halo_begin,
																				gm_y_vaild_begin, gm_y_vaild_end, gm_x_vaild_begin, gm_x_vaild_end, NY, NX);
			__syncwarp();
			sm_y_index_for_reg += core_y_tile_stream;
			sm_y_index_for_gm_input += core_y_tile_stream;
			sm_y_index_for_gm_output += core_y_tile_stream;
		}
	}
}

template<KernelType KERNEL_T, typename VALUE_T>
void FastStencil(VALUE_T *init_data, VALUE_T *output_data, int NY, int NX, int T, bool enable_warmup){
	int device, sm_count, numBlocksPerSm_current;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	const int CTX_SZ = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_CTX_SZ();
	const int BT = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_BT();
	const int sm_y_tile_mod = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_sm_y_tile_mod();
	const int gm_x_halo_tile = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_gm_x_halo_tile();
	const int gm_x_vaild_tile = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_gm_x_vaild_tile();

	const int grid_dim_x = (NX + gm_x_vaild_tile - 1) / gm_x_vaild_tile;
	const size_t sm_size = sizeof(VALUE_T)* (sm_y_tile_mod * gm_x_halo_tile);

	if(sm_size > props.sharedMemPerBlockOptin){
		LOG_ERROR("requested shared memory size (%lu bytes) exceeds device limit (%lu bytes)", sm_size, props.sharedMemPerBlockOptin);
		exit(1);
	}

	auto kernel_func = FastStencil_Kernel<KERNEL_T, VALUE_T>;
	cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, props.sharedMemPerBlockOptin);
	cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_current, kernel_func, CTX_SZ, sm_size);

	const int grid_dim_y = std::max(sm_count * numBlocksPerSm_current / grid_dim_x, 1);
	const int gm_y_vaild_tile = (NY + grid_dim_y - 1) / grid_dim_y;
	const int gm_y_halo_tile = gm_y_vaild_tile + RAD * BT * 2;

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

	if(enable_warmup){
		cudaEventRecord(start);
		kernel_func<<<dim3(grid_dim_x, grid_dim_y), CTX_SZ, sm_size>>>(gm_input, gm_output, NY, NX);
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
			kernel_func<<<dim3(grid_dim_x, grid_dim_y), CTX_SZ, sm_size>>>(gm_input, gm_output, NY, NX);
			cudaDeviceSynchronize();
		}
	}

	if(T % BT != 0){
		const int T_aligned = (T + BT - 1) / BT * BT;
		LOG_WARNING("T (%d) is not a multiple of BT (%d), adjusted to %d", T, BT, T_aligned);
		T = T_aligned;
	}

	cudaMemcpy(gm_input, init_data, data_memsize, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for(int t = 0; t < T; t+= BT){
		if(t != 0)
			std::swap(gm_input, gm_output);
		kernel_func<<<dim3(grid_dim_x, grid_dim_y), CTX_SZ, sm_size>>>(gm_input, gm_output, NY, NX);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
	double gstencils = 1.0 * NY * NX * T / elapsedTime / 1e6;
	double tflops = 1e-3 * gstencils * KernelInfo<KERNEL_T>::FP;
	double halo_rate = 1.0 * gm_x_vaild_tile / gm_x_halo_tile * gm_y_vaild_tile / gm_y_halo_tile;
	LOG_CRITICAL("%s: %lf GStencil/s, %lf TFLOPS, %.1lf%% Efficient computation(%.3lf/%.3lf)",
			  impl_name, gstencils, tflops, halo_rate * 100, tflops, tflops / halo_rate);

	cudaMemcpy(output_data, gm_output, data_memsize, cudaMemcpyDeviceToHost);

	cudaFree(gm_input);
	cudaFree(gm_output);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
template void FastStencil<KernelType::j2d5pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void FastStencil<KernelType::j2d9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void FastStencil<KernelType::j2ds9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void FastStencil<KernelType::j2d25pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void FastStencil<KernelType::j2d13pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
template void FastStencil<KernelType::j2d49pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
