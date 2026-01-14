#pragma once
#include "defines.h"

template<typename VALUE_T>
__device__ __forceinline__ VALUE_T FMA(VALUE_T a, VALUE_T b, VALUE_T c);
template<> __device__ __forceinline__ double FMA(double a, double b, double c){
	return __fma_rn(a, b, c);
}
template<> __device__ __forceinline__ float FMA(float a, float b, float c){
	return __fmaf_rn(a, b, c);
}
template<typename VALUE_T, int reg_z, int reg_z_end, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j3d7pt(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input_0)[reg_x_tile],
													 VALUE_T (*reg_input_1)[reg_x_tile], VALUE_T (*reg_input_2)[reg_x_tile]){
	constexpr VALUE_T w[][3][3] = W_j3d7pt;
	if constexpr (reg_z == 0){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[0][1][1] * reg_input_0[reg_y + 1][reg_x];
			}
		}
	} else if constexpr (reg_z == 1){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			reg_output[reg_y][0] = FMA(w[1][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][1], reg_input_1[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][2], reg_input_1[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][1], reg_input_1[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][0], reg_input_1[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][1], reg_input_1[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 2){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[2][1][1] * reg_input_2[reg_y + 1][reg_x];
			}
		}
	}
	if constexpr (reg_z + 1 < reg_z_end){
		kernelCompute_j3d7pt<VALUE_T, reg_z + 1, reg_z_end, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2);
	}
}
template<typename VALUE_T, int reg_z, int reg_z_end, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j3d13pt(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input_0)[reg_x_tile],
													  VALUE_T (*reg_input_1)[reg_x_tile], VALUE_T (*reg_input_2)[reg_x_tile],
													  VALUE_T (*reg_input_3)[reg_x_tile], VALUE_T (*reg_input_4)[reg_x_tile]){
	constexpr VALUE_T w[][5][5] = W_j3d13pt;
	if constexpr (reg_z == 0){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[0][2][2] * reg_input_0[reg_y + 2][reg_x];
			}
		}
	} else if constexpr (reg_z == 1){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[1][2][2] * reg_input_1[reg_y + 2][reg_x];
			}
		}
	} else if constexpr (reg_z == 2){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][0][2], reg_input_2[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][2], reg_input_2[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][4], reg_input_2[reg_y + 2][reg_x + 2], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 2] = FMA(w[2][2][4], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 2]);
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][2][4], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][1], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][3], reg_input_2[reg_y + 2][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][2][3], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][2], reg_input_2[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][2][1], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][1], reg_input_2[reg_y + 2][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][reg_x_tile - 2], 1), reg_output[reg_y][0]);
			reg_output[reg_y][1] = FMA(w[2][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][1]);
			#pragma unroll
			for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][0], reg_input_2[reg_y + 2][reg_x - 2], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][3][2], reg_input_2[reg_y + 3][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][4][2], reg_input_2[reg_y + 4][reg_x], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 3){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[3][2][2] * reg_input_3[reg_y + 2][reg_x];
			}
		}
	} else if constexpr (reg_z == 4){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] += w[4][2][2] * reg_input_4[reg_y + 2][reg_x];
			}
		}
	}
	if constexpr (reg_z + 1 < reg_z_end){
		kernelCompute_j3d13pt<VALUE_T, reg_z + 1, reg_z_end, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input_0, reg_input_1,
																							reg_input_2, reg_input_3, reg_input_4);
	}
}
template<typename VALUE_T, int reg_z, int reg_z_end, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_poisson(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input_0)[reg_x_tile],
													 VALUE_T (*reg_input_1)[reg_x_tile], VALUE_T (*reg_input_2)[reg_x_tile]){
	constexpr VALUE_T w[][3][3] = W_poisson;
	if constexpr (reg_z == 0){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][0][1], reg_input_0[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][2], reg_input_0[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[0][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_0[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][1], reg_input_0[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[0][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][0], reg_input_0[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][2][1], reg_input_0[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 1){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][2], reg_input_1[reg_y][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][0][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][1], reg_input_1[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][0][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][0], reg_input_1[reg_y][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][2], reg_input_1[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][1], reg_input_1[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][0], reg_input_1[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][2], reg_input_1[reg_y + 2][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][2][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][1], reg_input_1[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][0], reg_input_1[reg_y + 2][reg_x - 1], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 2){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][0][1], reg_input_2[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][2], reg_input_2[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][1], reg_input_2[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][0], reg_input_2[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][1], reg_input_2[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
		}
	}
	if constexpr (reg_z + 1 < reg_z_end){
		kernelCompute_poisson<VALUE_T, reg_z + 1, reg_z_end, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2);
	}
}
template<typename VALUE_T, int reg_z, int reg_z_end, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j3d27pt(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input_0)[reg_x_tile],
													 VALUE_T (*reg_input_1)[reg_x_tile], VALUE_T (*reg_input_2)[reg_x_tile]){
	constexpr VALUE_T w[][3][3] = W_j3d27pt;
	if constexpr (reg_z == 0){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][0][2], reg_input_0[reg_y][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[0][0][2], __shfl_down_sync(0xFFFFFFFF, reg_input_0[reg_y][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][0][1], reg_input_0[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[0][0][0], __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_y][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][0][0], reg_input_0[reg_y][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][2], reg_input_0[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[0][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_0[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][1], reg_input_0[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[0][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][1][0], reg_input_0[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][2][2], reg_input_0[reg_y + 2][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[0][2][2], __shfl_down_sync(0xFFFFFFFF, reg_input_0[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][2][1], reg_input_0[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[0][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[0][2][0], reg_input_0[reg_y + 2][reg_x - 1], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 1){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][2], reg_input_1[reg_y][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][0][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][1], reg_input_1[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][0][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][0][0], reg_input_1[reg_y][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][2], reg_input_1[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][1], reg_input_1[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][1][0], reg_input_1[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][2], reg_input_1[reg_y + 2][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[1][2][2], __shfl_down_sync(0xFFFFFFFF, reg_input_1[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][1], reg_input_1[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[1][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[1][2][0], reg_input_1[reg_y + 2][reg_x - 1], reg_output[reg_y][reg_x]);
			}
		}
	} else if constexpr (reg_z == 2){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][0][2], reg_input_2[reg_y][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][0][2], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][0][1], reg_input_2[reg_y][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][0][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][0][0], reg_input_2[reg_y][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][2], reg_input_2[reg_y + 1][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][1][2], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 1][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][1], reg_input_2[reg_y + 1][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][1][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 1][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][1][0], reg_input_2[reg_y + 1][reg_x - 1], reg_output[reg_y][reg_x]);
			}
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][2], reg_input_2[reg_y + 2][reg_x + 1], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][reg_x_tile - 1] = FMA(w[2][2][2], __shfl_down_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][0], 1), reg_output[reg_y][reg_x_tile - 1]);
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][1], reg_input_2[reg_y + 2][reg_x], reg_output[reg_y][reg_x]);
			}
			reg_output[reg_y][0] = FMA(w[2][2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_y + 2][reg_x_tile - 1], 1), reg_output[reg_y][0]);
			#pragma unroll
			for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
				reg_output[reg_y][reg_x] = FMA(w[2][2][0], reg_input_2[reg_y + 2][reg_x - 1], reg_output[reg_y][reg_x]);
			}
		}
	}
	if constexpr (reg_z + 1 < reg_z_end){
		kernelCompute_j3d27pt<VALUE_T, reg_z + 1, reg_z_end, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int kernel_z_begin, int kernel_z_end, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute(VALUE_T (*reg_output)[core_y_tile][reg_x_tile], VALUE_T (*reg_input)[reg_y_tile][reg_x_tile], int reg_z){
	if constexpr (kernel_z_begin == 0){
		#pragma unroll
		for(int reg_y = 0; reg_y < core_y_tile; reg_y++){
			#pragma unroll
			for(int reg_x = 0; reg_x < reg_x_tile; reg_x++)
				reg_output[reg_z][reg_y][reg_x] = 0;
		}
	}

	if constexpr (KERNEL_T == KernelType::j3d7pt){
		kernelCompute_j3d7pt<VALUE_T, kernel_z_begin, kernel_z_end, core_y_tile, reg_y_tile, reg_x_tile>
			(reg_output[reg_z], reg_input[reg_z], reg_input[reg_z + 1], reg_input[reg_z + 2]);
	}
	else if constexpr (KERNEL_T == KernelType::j3d13pt){
		kernelCompute_j3d13pt<VALUE_T, kernel_z_begin, kernel_z_end, core_y_tile, reg_y_tile, reg_x_tile>
			(reg_output[reg_z], reg_input[reg_z], reg_input[reg_z + 1], reg_input[reg_z + 2], reg_input[reg_z + 3], reg_input[reg_z + 4]);
	}
	else if constexpr (KERNEL_T == KernelType::poisson){
		kernelCompute_poisson<VALUE_T, kernel_z_begin, kernel_z_end, core_y_tile, reg_y_tile, reg_x_tile>
			(reg_output[reg_z], reg_input[reg_z], reg_input[reg_z + 1], reg_input[reg_z + 2]);
	}
	else if constexpr (KERNEL_T == KernelType::j3d27pt){
		kernelCompute_j3d27pt<VALUE_T, kernel_z_begin, kernel_z_end, core_y_tile, reg_y_tile, reg_x_tile>
			(reg_output[reg_z], reg_input[reg_z], reg_input[reg_z + 1], reg_input[reg_z + 2]);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int core_z_tile_stream, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute(VALUE_T (*reg_output)[core_y_tile][reg_x_tile], VALUE_T (*reg_input)[reg_y_tile][reg_x_tile]){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	#pragma unroll
	for(int reg_z = 0; reg_z < core_z_tile_stream; reg_z++){
		kernelCompute<KERNEL_T, VALUE_T, 0, RAD * 2 + 1, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input, reg_z);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int core_z_tile_stream, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_pre(VALUE_T (*reg_output)[core_y_tile][reg_x_tile], VALUE_T (*reg_input)[reg_y_tile][reg_x_tile]){
	kernelCompute<KERNEL_T, VALUE_T, 0, 1, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input, 0);
}
template<KernelType KERNEL_T, typename VALUE_T, int core_z_tile_stream, int core_y_tile, int reg_y_tile, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_post(VALUE_T (*reg_output)[core_y_tile][reg_x_tile], VALUE_T (*reg_input)[reg_y_tile][reg_x_tile]){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	kernelCompute<KERNEL_T, VALUE_T, 1, RAD * 2 + 1, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input, 0);
	#pragma unroll
	for(int reg_z = 1; reg_z < core_z_tile_stream; reg_z++){
		kernelCompute<KERNEL_T, VALUE_T, 0, RAD * 2 + 1, core_y_tile, reg_y_tile, reg_x_tile>(reg_output, reg_input, reg_z);
	}
}
