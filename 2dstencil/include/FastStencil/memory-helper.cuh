#pragma once
#include <cuda_pipeline_primitives.h>
#include "defines.h"
#define Offset_sm_mod(i,j) ((j) + sm_x_tile * ((i) & (sm_y_tile_mod - 1)))

template<typename VALUE_T, int data_y_length, int reg_y_begin, int sm_y_tile_mod, int reg_x_tile, int sm_x_tile>
__device__ __forceinline__ void sm2reg(VALUE_T (*reg_data)[reg_x_tile], VALUE_T *sm_data, int sm_y_begin){
	const int lane_idx = threadIdx.x % 32;
	#pragma unroll
	for(int data_y = 0; data_y < data_y_length; data_y++){
		int sm_y = sm_y_begin + data_y;
		int reg_y = reg_y_begin + data_y;
		#pragma unroll
		for(int data_x = 0; data_x < reg_x_tile; data_x++){
			int sm_x = reg_x_tile * lane_idx + data_x;
			int reg_x = data_x;
			reg_data[reg_y][reg_x] = sm_data[Offset_sm_mod(sm_y, sm_x)];
		}
	}
}
template<typename VALUE_T, int data_y_length, int reg_y_begin, int sm_y_tile_mod, int reg_x_tile, int sm_x_tile>
__device__ __forceinline__ void reg2sm(VALUE_T *sm_data, VALUE_T (*reg_data)[reg_x_tile], int sm_y_begin){
	const int lane_idx = threadIdx.x % 32;
	#pragma unroll
	for(int data_y = 0; data_y < data_y_length; data_y++){
		int sm_y = sm_y_begin + data_y;
		int reg_y = reg_y_begin + data_y;
		#pragma unroll
		for(int data_x = 0; data_x < reg_x_tile; data_x++){
			int sm_x = reg_x_tile * lane_idx + data_x;
			int reg_x = data_x;
			sm_data[Offset_sm_mod(sm_y, sm_x)] = reg_data[reg_y][reg_x];
		}
	}
}
template<typename VALUE_T, int data_y_length, int reg_dest_y_begin, int reg_src_y_begin, int reg_x_tile>
__device__ __forceinline__ void reg2reg(VALUE_T (*reg_dest)[reg_x_tile], VALUE_T (*reg_src)[reg_x_tile]){
	#pragma unroll
	for(int data_y = 0; data_y < data_y_length; data_y++){
		#pragma unroll
		for(int data_x = 0; data_x < reg_x_tile; data_x++){
			reg_dest[reg_dest_y_begin + data_y][data_x] = reg_src[reg_src_y_begin + data_y][data_x];
		}
	}
}
template<typename VALUE_T, int CTX_SZ, int data_y_length, int sm_y_tile_mod, int data_x_length, int sm_x_tile>
__device__ __forceinline__ void gm2sm(VALUE_T *sm_data, VALUE_T *gm_input, int sm_y_begin, int gm_y_begin, int gm_x_begin, int NY, int NX){
	constexpr int data_x_length_thread = data_x_length / CTX_SZ;
	#pragma unroll
	for(int data_y = 0; data_y < data_y_length; data_y++){
		int sm_y = sm_y_begin + data_y;
		int gm_y = min(max(gm_y_begin + data_y, 0), NY - 1);
		#pragma unroll
		for(int data_x = 0; data_x < data_x_length_thread; data_x++){
			int sm_x = data_x * CTX_SZ + threadIdx.x;
			int gm_x = min(max(gm_x_begin + sm_x, 0), NX - 1);
			__pipeline_memcpy_async(&sm_data[Offset_sm_mod(sm_y, sm_x)], &gm_input[Offset_gm(gm_y, gm_x)], sizeof(VALUE_T));
		}
		if(threadIdx.x < data_x_length % CTX_SZ){
			int data_x = data_x_length_thread;
			int sm_x = data_x * CTX_SZ + threadIdx.x;
			int gm_x = min(max(gm_x_begin + sm_x, 0), NX - 1);
			__pipeline_memcpy_async(&sm_data[Offset_sm_mod(sm_y, sm_x)], &gm_input[Offset_gm(gm_y, gm_x)], sizeof(VALUE_T));
		}
	}
	__pipeline_commit();
}
template<typename VALUE_T, int CTX_SZ, int data_y_length, int sm_y_tile_mod, int data_x_length, int sm_x_tile>
__device__ __forceinline__ void sm2gm(VALUE_T *gm_output, VALUE_T *sm_data, int sm_y_begin, int gm_y_begin, int gm_x_begin,
													  int gm_y_vaild_begin, int gm_y_vaild_end, int gm_x_vaild_begin, int gm_x_vaild_end, int NY, int NX){
	constexpr int data_x_length_thread = data_x_length / CTX_SZ;
	#pragma unroll
	for(int data_y = 0; data_y < data_y_length; data_y++){
		int sm_y = sm_y_begin + data_y;
		int gm_y = gm_y_begin + data_y;
		#pragma unroll
		for(int data_x = 0; data_x < data_x_length_thread; data_x++){
			int sm_x = data_x * CTX_SZ + threadIdx.x;
			int gm_x = gm_x_begin + sm_x;
			if(gm_y >= gm_y_vaild_begin && gm_y < gm_y_vaild_end && gm_x >= gm_x_vaild_begin && gm_x < gm_x_vaild_end){
				gm_output[Offset_gm(gm_y, gm_x)] = sm_data[Offset_sm_mod(sm_y, sm_x)];
			}
		}
		if(threadIdx.x < data_x_length % CTX_SZ){
			int data_x = data_x_length_thread;
			int sm_x = data_x * CTX_SZ + threadIdx.x;
			int gm_x = gm_x_begin + sm_x;
			if(gm_y >= gm_y_vaild_begin && gm_y < gm_y_vaild_end && gm_x >= gm_x_vaild_begin && gm_x < gm_x_vaild_end){
				gm_output[Offset_gm(gm_y, gm_x)] = sm_data[Offset_sm_mod(sm_y, sm_x)];
			}
		}
	}
}
