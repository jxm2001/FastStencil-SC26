#pragma once
#include <cuda_pipeline_primitives.h>
#include "defines.h"
#define Offset_sm_mod(i,j,k) ((k) + sm_x_tile * ((j) + sm_y_tile * ((i) & (sm_z_tile_mod - 1))))
#define Offset_sm(i,j,k) ((k) + sm_x_tile * ((j) + sm_y_tile * (i)))

template<typename VALUE_T, int data_z_length, int reg_z_begin, int sm_z_tile_mod,
int data_y_length, int reg_y_begin, int reg_y_tile, int sm_y_tile, int reg_x_tile, int sm_x_tile, bool enable_mod>
__device__ __forceinline__ void sm2reg(VALUE_T (*reg_data)[reg_y_tile][reg_x_tile], VALUE_T *sm_data, int sm_z_begin, int sm_y_begin){
	const int lane_idx = threadIdx.x % 32;
	#pragma unroll
	for(int data_z = 0; data_z < data_z_length; data_z++){
		int sm_z = sm_z_begin + data_z;
		int reg_z = reg_z_begin + data_z;
		#pragma unroll
		for(int data_y = 0; data_y < data_y_length; data_y++){
			int sm_y = sm_y_begin + data_y;
			int reg_y = reg_y_begin + data_y;
			#pragma unroll
			for(int data_x = 0; data_x < reg_x_tile; data_x++){
				int sm_x = reg_x_tile * lane_idx + data_x;
				int reg_x = data_x;
				if constexpr(enable_mod)
					reg_data[reg_z][reg_y][reg_x] = sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)];
				else
					reg_data[reg_z][reg_y][reg_x] = sm_data[Offset_sm(sm_z, sm_y, sm_x)];
			}
		}
	}
}
template<typename VALUE_T, int data_z_length, int reg_z_begin, int sm_z_tile_mod,
int data_y_length, int reg_y_begin, int reg_y_tile, int sm_y_tile, int reg_x_tile, int sm_x_tile, bool enable_mod>
__device__ __forceinline__ void reg2sm(VALUE_T *sm_data, VALUE_T (*reg_data)[reg_y_tile][reg_x_tile], int sm_z_begin, int sm_y_begin){
	const int lane_idx = threadIdx.x % 32;
	#pragma unroll
	for(int data_z = 0; data_z < data_z_length; data_z++){
		int sm_z = sm_z_begin + data_z;
		int reg_z = reg_z_begin + data_z;
		#pragma unroll
		for(int data_y = 0; data_y < data_y_length; data_y++){
			int sm_y = sm_y_begin + data_y;
			int reg_y = reg_y_begin + data_y;
			#pragma unroll
			for(int data_x = 0; data_x < reg_x_tile; data_x++){
				int sm_x = reg_x_tile * lane_idx + data_x;
				int reg_x = data_x;
				if constexpr(enable_mod)
					sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)] = reg_data[reg_z][reg_y][reg_x];
				else
					sm_data[Offset_sm(sm_z, sm_y, sm_x)] = reg_data[reg_z][reg_y][reg_x];
			}
		}
	}
}
template<typename VALUE_T, int data_z_length, int reg_dest_z_begin, int reg_src_z_begin,
int data_y_length, int reg_dest_y_begin, int reg_src_y_begin, int reg_dest_y_tile, int reg_src_y_tile, int reg_x_tile>
__device__ __forceinline__ void reg2reg(VALUE_T (*reg_dest)[reg_dest_y_tile][reg_x_tile], VALUE_T (*reg_src)[reg_src_y_tile][reg_x_tile]){
	#pragma unroll
	for(int data_z = 0; data_z < data_z_length; data_z++){
		#pragma unroll
		for(int data_y = 0; data_y < data_y_length; data_y++){
			#pragma unroll
			for(int data_x = 0; data_x < reg_x_tile; data_x++){
				reg_dest[reg_dest_z_begin + data_z][reg_dest_y_begin + data_y][data_x] = reg_src[reg_src_z_begin + data_z][reg_src_y_begin + data_y][data_x];
			}
		}
	}
}
template<typename VALUE_T, int warps_num, int data_z_length, int sm_z_tile_mod, int sm_y_tile, int sm_x_tile>
__device__ __forceinline__ void gm2sm(VALUE_T *sm_data, VALUE_T *gm_input, int sm_z_begin, int gm_z_begin, int gm_y_begin, int gm_x_begin,
									  int NZ, int NY, int NX){
	constexpr int data_yz_length = data_z_length * sm_y_tile;
	constexpr int data_yz_length_thread = data_yz_length / warps_num * warps_num;
	const int warp_idx = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
	const int lane_idx = threadIdx.x % 32;
	#pragma unroll
	for(int data_yz = 0; data_yz < data_yz_length_thread; data_yz += warps_num){
		int data_z = (data_yz + warp_idx) / sm_y_tile;
		int sm_z = sm_z_begin + data_z;
		int gm_z = min(max(gm_z_begin + data_z, 0), NZ - 1);
		int sm_y = (data_yz + warp_idx) % sm_y_tile;
		int gm_y = min(max(gm_y_begin + sm_y, 0), NY - 1);
		#pragma unroll
		for(int data_x = 0; data_x < sm_x_tile; data_x += 32){
			int sm_x = data_x + lane_idx;
			int gm_x = min(max(gm_x_begin + sm_x, 0), NX - 1);
			__pipeline_memcpy_async(&sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)], &gm_input[Offset_gm(gm_z, gm_y, gm_x)], sizeof(VALUE_T));
		}
	}
	if (warp_idx < data_yz_length % warps_num){
		int data_yz = data_yz_length_thread;
		int data_z = (data_yz + warp_idx) / sm_y_tile;
		int sm_z = sm_z_begin + data_z;
		int gm_z = min(max(gm_z_begin + data_z, 0), NZ - 1);
		int sm_y = (data_yz + warp_idx) % sm_y_tile;
		int gm_y = min(max(gm_y_begin + sm_y, 0), NY - 1);
		#pragma unroll
		for(int data_x = 0; data_x < sm_x_tile; data_x += 32){
			int sm_x = data_x + lane_idx;
			int gm_x = min(max(gm_x_begin + sm_x, 0), NX - 1);
			__pipeline_memcpy_async(&sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)], &gm_input[Offset_gm(gm_z, gm_y, gm_x)], sizeof(VALUE_T));
		}
	}
	__pipeline_commit();
}
template<typename VALUE_T, int warps_num, int data_z_length, int sm_z_tile_mod, int sm_y_tile, int sm_x_tile>
__device__ __forceinline__ void sm2gm(VALUE_T *gm_output, VALUE_T *sm_data, int sm_z_begin, int gm_z_begin, int gm_y_begin, int gm_x_begin,
									  int gm_z_vaild_begin, int gm_z_vaild_end, int gm_y_vaild_begin, int gm_y_vaild_end,
									  int gm_x_vaild_begin, int gm_x_vaild_end, int NZ, int NY, int NX){
	constexpr int data_yz_length = data_z_length * sm_y_tile;
	constexpr int data_yz_length_thread = data_yz_length / warps_num * warps_num;
	const int warp_idx = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
	const int lane_idx = threadIdx.x % 32;
	if (warp_idx < data_yz_length % warps_num){
		int data_yz = data_yz_length_thread;
		int data_z = (data_yz + warp_idx) / sm_y_tile;
		int sm_z = sm_z_begin + data_z;
		int gm_z = gm_z_begin + data_z;
		int sm_y = (data_yz + warp_idx) % sm_y_tile;
		int gm_y = gm_y_begin + sm_y;
		#pragma unroll
		for(int data_x = 0; data_x < sm_x_tile; data_x += 32){
			int sm_x = data_x + lane_idx;
			int gm_x = gm_x_begin + sm_x;
			if(gm_z >= gm_z_vaild_begin && gm_z < gm_z_vaild_end && gm_y >= gm_y_vaild_begin && gm_y < gm_y_vaild_end &&
				gm_x >= gm_x_vaild_begin && gm_x < gm_x_vaild_end){
				gm_output[Offset_gm(gm_z, gm_y, gm_x)] = sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)];
			}
		}
	}
	#pragma unroll
	for(int data_yz = 0; data_yz < data_yz_length_thread; data_yz += warps_num){
		int data_z = (data_yz + warp_idx) / sm_y_tile;
		int sm_z = sm_z_begin + data_z;
		int gm_z = gm_z_begin + data_z;
		int sm_y = (data_yz + warp_idx) % sm_y_tile;
		int gm_y = gm_y_begin + sm_y;
		#pragma unroll
		for(int data_x = 0; data_x < sm_x_tile; data_x += 32){
			int sm_x = data_x + lane_idx;
			int gm_x = gm_x_begin + sm_x;
			if(gm_z >= gm_z_vaild_begin && gm_z < gm_z_vaild_end && gm_y >= gm_y_vaild_begin && gm_y < gm_y_vaild_end &&
				gm_x >= gm_x_vaild_begin && gm_x < gm_x_vaild_end){
				gm_output[Offset_gm(gm_z, gm_y, gm_x)] = sm_data[Offset_sm_mod(sm_z, sm_y, sm_x)];
			}
		}
	}
}
