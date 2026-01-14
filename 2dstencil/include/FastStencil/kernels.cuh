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
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2d5pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2){
	constexpr VALUE_T w[][3] = W_j2d5pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][1] * reg_input_0[reg_x];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[1][2] * reg_input_1[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[1][2] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][1] * reg_input_1[reg_x];
		}
		reg_output[0] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][0] * reg_input_1[reg_x - 1];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][1] * reg_input_2[reg_x];
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2d5pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2);
	}
}
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2d9pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2){
	constexpr VALUE_T w[][3] = W_j2d9pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[0][2] * reg_input_0[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[0][2] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][1] * reg_input_0[reg_x];
		}
		reg_output[0] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][0] * reg_input_0[reg_x - 1];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[1][2] * reg_input_1[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[1][2] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][1] * reg_input_1[reg_x];
		}
		reg_output[0] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][0] * reg_input_1[reg_x - 1];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] = FMA(w[2][2], reg_input_2[reg_x + 1], reg_output[reg_x]);
		}
		reg_output[reg_x_tile - 1] = FMA(w[2][2], __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1), reg_output[reg_x_tile - 1]);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[2][1], reg_input_2[reg_x], reg_output[reg_x]);
		}
		reg_output[0] = FMA(w[2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1), reg_output[0]);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[2][0], reg_input_2[reg_x - 1], reg_output[reg_x]);
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2d9pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2);
	}
}
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2ds9pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2, VALUE_T *reg_input_3, VALUE_T *reg_input_4){
	constexpr VALUE_T w[][5] = W_j2ds9pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][2] * reg_input_0[reg_x];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][2] * reg_input_1[reg_x];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[2][4] * reg_input_2[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[2][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1);
		reg_output[reg_x_tile - 1] += w[2][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[2][3] * reg_input_2[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[2][3] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][2] * reg_input_2[reg_x];
		}
		reg_output[0] += w[2][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][1] * reg_input_2[reg_x - 1];
		}
		reg_output[0] += w[2][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 2], 1);
		reg_output[1] += w[2][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][0] * reg_input_2[reg_x - 2];
		}
	} else if constexpr (kernel_y == 3){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][2] * reg_input_3[reg_x];
		}
	} else if constexpr (kernel_y == 4){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][2] * reg_input_4[reg_x];
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2ds9pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2, reg_input_3, reg_input_4);
	}
}
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2d25pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2, VALUE_T *reg_input_3, VALUE_T *reg_input_4){
	constexpr VALUE_T w[][5] = W_j2d25pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[0][4] * reg_input_0[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[0][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		reg_output[reg_x_tile - 1] += w[0][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[0][3] * reg_input_0[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[0][3] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][2] * reg_input_0[reg_x];
		}
		reg_output[0] += w[0][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][1] * reg_input_0[reg_x - 1];
		}
		reg_output[0] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 2], 1);
		reg_output[1] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][0] * reg_input_0[reg_x - 2];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[1][4] * reg_input_1[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[1][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		reg_output[reg_x_tile - 1] += w[1][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[1][3] * reg_input_1[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[1][3] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][2] * reg_input_1[reg_x];
		}
		reg_output[0] += w[1][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][1] * reg_input_1[reg_x - 1];
		}
		reg_output[0] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 2], 1);
		reg_output[1] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][0] * reg_input_1[reg_x - 2];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] = FMA(w[2][4], reg_input_2[reg_x + 2], reg_output[reg_x]);
		}
		reg_output[reg_x_tile - 2] = FMA(w[2][4], __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1), reg_output[reg_x_tile - 2]);
		reg_output[reg_x_tile - 1] = FMA(w[2][4], __shfl_down_sync(0xFFFFFFFF, reg_input_2[1], 1), reg_output[reg_x_tile - 1]);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] = FMA(w[2][3], reg_input_2[reg_x + 1], reg_output[reg_x]);
		}
		reg_output[reg_x_tile - 1] = FMA(w[2][3], __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1), reg_output[reg_x_tile - 1]);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[2][2], reg_input_2[reg_x], reg_output[reg_x]);
		}
		reg_output[0] = FMA(w[2][1], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1), reg_output[0]);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[2][1], reg_input_2[reg_x - 1], reg_output[reg_x]);
		}
		reg_output[0] = FMA(w[2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 2], 1), reg_output[0]);
		reg_output[1] = FMA(w[2][0], __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1), reg_output[1]);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[2][0], reg_input_2[reg_x - 2], reg_output[reg_x]);
		}
	} else if constexpr (kernel_y == 3){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[3][4] * reg_input_3[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[3][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		reg_output[reg_x_tile - 1] += w[3][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[3][3] * reg_input_3[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[3][3] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][2] * reg_input_3[reg_x];
		}
		reg_output[0] += w[3][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][1] * reg_input_3[reg_x - 1];
		}
		reg_output[0] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 2], 1);
		reg_output[1] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][0] * reg_input_3[reg_x - 2];
		}
	} else if constexpr (kernel_y == 4){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] = FMA(w[4][4], reg_input_4[reg_x + 2], reg_output[reg_x]);
		}
		reg_output[reg_x_tile - 2] = FMA(w[4][4], __shfl_down_sync(0xFFFFFFFF, reg_input_4[0], 1), reg_output[reg_x_tile - 2]);
		reg_output[reg_x_tile - 1] = FMA(w[4][4], __shfl_down_sync(0xFFFFFFFF, reg_input_4[1], 1), reg_output[reg_x_tile - 1]);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] = FMA(w[4][3], reg_input_4[reg_x + 1], reg_output[reg_x]);
		}
		reg_output[reg_x_tile - 1] = FMA(w[4][3], __shfl_down_sync(0xFFFFFFFF, reg_input_4[0], 1), reg_output[reg_x_tile - 1]);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[4][2], reg_input_4[reg_x], reg_output[reg_x]);
		}
		reg_output[0] = FMA(w[4][1], __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 1], 1), reg_output[0]);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[4][1], reg_input_4[reg_x - 1], reg_output[reg_x]);
		}
		reg_output[0] = FMA(w[4][0], __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 2], 1), reg_output[0]);
		reg_output[1] = FMA(w[4][0], __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 1], 1), reg_output[1]);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] = FMA(w[4][0], reg_input_4[reg_x - 2], reg_output[reg_x]);
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2d25pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2, reg_input_3, reg_input_4);
	}
}
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2d13pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2,
													  VALUE_T *reg_input_3, VALUE_T *reg_input_4, VALUE_T *reg_input_5, VALUE_T *reg_input_6){
	constexpr VALUE_T w[][7] = W_j2d13pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][3] * reg_input_0[reg_x];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][3] * reg_input_1[reg_x];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][3] * reg_input_2[reg_x];
		}
	} else if constexpr (kernel_y == 3){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[3][6] * reg_input_3[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		reg_output[reg_x_tile - 2] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[1], 1);
		reg_output[reg_x_tile - 1] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[3][5] * reg_input_3[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[3][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		reg_output[reg_x_tile - 1] += w[3][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[3][4] * reg_input_3[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[3][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][3] * reg_input_3[reg_x];
		}
		reg_output[0] += w[3][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][2] * reg_input_3[reg_x - 1];
		}
		reg_output[0] += w[3][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 2], 1);
		reg_output[1] += w[3][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][1] * reg_input_3[reg_x - 2];
		}
		reg_output[0] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 3], 1);
		reg_output[1] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 2], 1);
		reg_output[2] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][0] * reg_input_3[reg_x - 3];
		}
	} else if constexpr (kernel_y == 4){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][3] * reg_input_4[reg_x];
		}
	} else if constexpr (kernel_y == 5){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[5][3] * reg_input_5[reg_x];
		}
	} else if constexpr (kernel_y == 6){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[6][3] * reg_input_6[reg_x];
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2d13pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2,
																		 reg_input_3, reg_input_4, reg_input_5, reg_input_6);
	}
}
template<typename VALUE_T, int kernel_y, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_j2d49pt(VALUE_T *reg_output, VALUE_T *reg_input_0, VALUE_T *reg_input_1, VALUE_T *reg_input_2,
													  VALUE_T *reg_input_3, VALUE_T *reg_input_4, VALUE_T *reg_input_5, VALUE_T *reg_input_6){
	constexpr VALUE_T w[][7] = W_j2d49pt;
	if constexpr (kernel_y == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[0][6] * reg_input_0[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[0][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		reg_output[reg_x_tile - 2] += w[0][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[1], 1);
		reg_output[reg_x_tile - 1] += w[0][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[0][5] * reg_input_0[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[0][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		reg_output[reg_x_tile - 1] += w[0][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[0][4] * reg_input_0[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[0][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_0[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][3] * reg_input_0[reg_x];
		}
		reg_output[0] += w[0][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][2] * reg_input_0[reg_x - 1];
		}
		reg_output[0] += w[0][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 2], 1);
		reg_output[1] += w[0][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][1] * reg_input_0[reg_x - 2];
		}
		reg_output[0] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 3], 1);
		reg_output[1] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 2], 1);
		reg_output[2] += w[0][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_0[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[0][0] * reg_input_0[reg_x - 3];
		}
	} else if constexpr (kernel_y == 1){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[1][6] * reg_input_1[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[1][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		reg_output[reg_x_tile - 2] += w[1][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[1], 1);
		reg_output[reg_x_tile - 1] += w[1][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[1][5] * reg_input_1[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[1][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		reg_output[reg_x_tile - 1] += w[1][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[1][4] * reg_input_1[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[1][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_1[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][3] * reg_input_1[reg_x];
		}
		reg_output[0] += w[1][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][2] * reg_input_1[reg_x - 1];
		}
		reg_output[0] += w[1][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 2], 1);
		reg_output[1] += w[1][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][1] * reg_input_1[reg_x - 2];
		}
		reg_output[0] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 3], 1);
		reg_output[1] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 2], 1);
		reg_output[2] += w[1][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_1[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[1][0] * reg_input_1[reg_x - 3];
		}
	} else if constexpr (kernel_y == 2){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[2][6] * reg_input_2[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[2][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1);
		reg_output[reg_x_tile - 2] += w[2][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[1], 1);
		reg_output[reg_x_tile - 1] += w[2][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[2][5] * reg_input_2[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[2][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1);
		reg_output[reg_x_tile - 1] += w[2][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[2][4] * reg_input_2[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[2][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_2[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][3] * reg_input_2[reg_x];
		}
		reg_output[0] += w[2][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][2] * reg_input_2[reg_x - 1];
		}
		reg_output[0] += w[2][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 2], 1);
		reg_output[1] += w[2][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][1] * reg_input_2[reg_x - 2];
		}
		reg_output[0] += w[2][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 3], 1);
		reg_output[1] += w[2][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 2], 1);
		reg_output[2] += w[2][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_2[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[2][0] * reg_input_2[reg_x - 3];
		}
	} else if constexpr (kernel_y == 3){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[3][6] * reg_input_3[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		reg_output[reg_x_tile - 2] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[1], 1);
		reg_output[reg_x_tile - 1] += w[3][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[3][5] * reg_input_3[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[3][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		reg_output[reg_x_tile - 1] += w[3][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[3][4] * reg_input_3[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[3][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_3[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][3] * reg_input_3[reg_x];
		}
		reg_output[0] += w[3][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][2] * reg_input_3[reg_x - 1];
		}
		reg_output[0] += w[3][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 2], 1);
		reg_output[1] += w[3][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][1] * reg_input_3[reg_x - 2];
		}
		reg_output[0] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 3], 1);
		reg_output[1] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 2], 1);
		reg_output[2] += w[3][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_3[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[3][0] * reg_input_3[reg_x - 3];
		}
	} else if constexpr (kernel_y == 4){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[4][6] * reg_input_4[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[4][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[0], 1);
		reg_output[reg_x_tile - 2] += w[4][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[1], 1);
		reg_output[reg_x_tile - 1] += w[4][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[4][5] * reg_input_4[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[4][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[0], 1);
		reg_output[reg_x_tile - 1] += w[4][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[4][4] * reg_input_4[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[4][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_4[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][3] * reg_input_4[reg_x];
		}
		reg_output[0] += w[4][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][2] * reg_input_4[reg_x - 1];
		}
		reg_output[0] += w[4][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 2], 1);
		reg_output[1] += w[4][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][1] * reg_input_4[reg_x - 2];
		}
		reg_output[0] += w[4][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 3], 1);
		reg_output[1] += w[4][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 2], 1);
		reg_output[2] += w[4][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_4[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[4][0] * reg_input_4[reg_x - 3];
		}
	} else if constexpr (kernel_y == 5){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[5][6] * reg_input_5[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[5][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[0], 1);
		reg_output[reg_x_tile - 2] += w[5][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[1], 1);
		reg_output[reg_x_tile - 1] += w[5][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[5][5] * reg_input_5[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[5][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[0], 1);
		reg_output[reg_x_tile - 1] += w[5][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[5][4] * reg_input_5[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[5][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_5[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[5][3] * reg_input_5[reg_x];
		}
		reg_output[0] += w[5][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[5][2] * reg_input_5[reg_x - 1];
		}
		reg_output[0] += w[5][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 2], 1);
		reg_output[1] += w[5][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[5][1] * reg_input_5[reg_x - 2];
		}
		reg_output[0] += w[5][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 3], 1);
		reg_output[1] += w[5][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 2], 1);
		reg_output[2] += w[5][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_5[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[5][0] * reg_input_5[reg_x - 3];
		}
	} else if constexpr (kernel_y == 6){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 3; reg_x++){
			reg_output[reg_x] += w[6][6] * reg_input_6[reg_x + 3];
		}
		reg_output[reg_x_tile - 3] += w[6][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[0], 1);
		reg_output[reg_x_tile - 2] += w[6][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[1], 1);
		reg_output[reg_x_tile - 1] += w[6][6] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[2], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 2; reg_x++){
			reg_output[reg_x] += w[6][5] * reg_input_6[reg_x + 2];
		}
		reg_output[reg_x_tile - 2] += w[6][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[0], 1);
		reg_output[reg_x_tile - 1] += w[6][5] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[1], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile - 1; reg_x++){
			reg_output[reg_x] += w[6][4] * reg_input_6[reg_x + 1];
		}
		reg_output[reg_x_tile - 1] += w[6][4] * __shfl_down_sync(0xFFFFFFFF, reg_input_6[0], 1);
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[6][3] * reg_input_6[reg_x];
		}
		reg_output[0] += w[6][2] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 1; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[6][2] * reg_input_6[reg_x - 1];
		}
		reg_output[0] += w[6][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 2], 1);
		reg_output[1] += w[6][1] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 2; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[6][1] * reg_input_6[reg_x - 2];
		}
		reg_output[0] += w[6][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 3], 1);
		reg_output[1] += w[6][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 2], 1);
		reg_output[2] += w[6][0] * __shfl_up_sync(0xFFFFFFFF, reg_input_6[reg_x_tile - 1], 1);
		#pragma unroll
		for(int reg_x = 3; reg_x < reg_x_tile; reg_x++){
			reg_output[reg_x] += w[6][0] * reg_input_6[reg_x - 3];
		}
	}
	if constexpr (kernel_y + 1 < kernel_y_end){
		kernelCompute_j2d49pt<VALUE_T, kernel_y + 1, kernel_y_end, reg_x_tile>(reg_output, reg_input_0, reg_input_1, reg_input_2,
																		 reg_input_3, reg_input_4, reg_input_5, reg_input_6);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int kernel_y_begin, int kernel_y_end, int reg_x_tile>
__device__ __forceinline__ void kernelCompute(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input)[reg_x_tile], int reg_y){
	if constexpr (kernel_y_begin == 0){
		#pragma unroll
		for(int reg_x = 0; reg_x < reg_x_tile; reg_x++)
			reg_output[reg_y][reg_x] = 0;
	}

	if constexpr (KERNEL_T == KernelType::j2d5pt){
		kernelCompute_j2d5pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2]);
	}
	else if constexpr (KERNEL_T == KernelType::j2d9pt){
		kernelCompute_j2d9pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2]);
	}
	else if constexpr (KERNEL_T == KernelType::j2ds9pt){
		kernelCompute_j2ds9pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2],
																	 reg_input[reg_y + 3], reg_input[reg_y + 4]);
	}
	else if constexpr (KERNEL_T == KernelType::j2d25pt){
		kernelCompute_j2d25pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2],
																	 reg_input[reg_y + 3], reg_input[reg_y + 4]);
	}
	else if constexpr (KERNEL_T == KernelType::j2d13pt){
		kernelCompute_j2d13pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2],
																	 reg_input[reg_y + 3], reg_input[reg_y + 4], reg_input[reg_y + 5], reg_input[reg_y + 6]);
	}
	else{
		kernelCompute_j2d49pt<VALUE_T, kernel_y_begin, kernel_y_end, reg_x_tile>(reg_output[reg_y], reg_input[reg_y], reg_input[reg_y + 1], reg_input[reg_y + 2],
																	 reg_input[reg_y + 3], reg_input[reg_y + 4], reg_input[reg_y + 5], reg_input[reg_y + 6]);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int core_y_tile_stream, int reg_x_tile>
__device__ __forceinline__ void kernelCompute(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input)[reg_x_tile]){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	#pragma unroll
	for(int reg_y = 0; reg_y < core_y_tile_stream; reg_y++){
		kernelCompute<KERNEL_T, VALUE_T, 0, RAD * 2 + 1, reg_x_tile>(reg_output, reg_input, reg_y);
	}
}
template<KernelType KERNEL_T, typename VALUE_T, int core_y_tile_stream, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_pre(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input)[reg_x_tile]){
	kernelCompute<KERNEL_T, VALUE_T, 0, 1, reg_x_tile>(reg_output, reg_input, 0);
}
template<KernelType KERNEL_T, typename VALUE_T, int core_y_tile_stream, int reg_x_tile>
__device__ __forceinline__ void kernelCompute_post(VALUE_T (*reg_output)[reg_x_tile], VALUE_T (*reg_input)[reg_x_tile]){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	kernelCompute<KERNEL_T, VALUE_T, 1, RAD * 2 + 1, reg_x_tile>(reg_output, reg_input, 0);
	#pragma unroll
	for(int reg_y = 1; reg_y < core_y_tile_stream; reg_y++){
		kernelCompute<KERNEL_T, VALUE_T, 0, RAD * 2 + 1, reg_x_tile>(reg_output, reg_input, reg_y);
	}
}
