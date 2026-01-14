#pragma once
#define Offset_gm(i,j,k) ((k) + NX * (j + NY * (i)))

enum class KernelType{
	j3d7pt,
	j3d13pt,
	poisson,
	j3d27pt,
	unknow
};

template<KernelType KERNEL_T>
struct KernelInfo;

template <>
struct KernelInfo<KernelType::j3d7pt>{
	static constexpr int RAD = 1;
	static constexpr int FP = 14;
};
#define W_j3d7pt { \
	{ \
		{   0,    0,    0}, \
		{   0, 0.01,    0}, \
		{   0,    0,    0}, \
	}, \
	{ \
		{   0, 0.02,    0}, \
		{0.03, 0.79, 0.04}, \
		{   0, 0.05,    0}, \
	}, \
	{ \
		{   0,    0,    0}, \
		{   0, 0.06,    0}, \
		{   0,    0,    0}, \
	}, \
}
template <>
struct KernelInfo<KernelType::j3d13pt>{
	static constexpr int RAD = 2;
	static constexpr int FP = 26;
};
#define W_j3d13pt { \
	{ \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0, 0.01,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
	}, \
	{ \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0, 0.02,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
	}, \
	{ \
		{   0,    0, 0.03,    0,    0}, \
		{   0,    0, 0.04,    0,    0}, \
		{0.05, 0.06, 0.22, 0.07, 0.08}, \
		{   0,    0, 0.09,    0,    0}, \
		{   0,    0, 0.10,    0,    0}, \
	}, \
	{ \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0, 0.11,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
	}, \
	{ \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0, 0.12,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
		{   0,    0,    0,    0,    0}, \
	}, \
}
template <>
struct KernelInfo<KernelType::poisson>{
	static constexpr int RAD = 1;
	static constexpr int FP = 38;
};
#define W_poisson { \
	{ \
		{      0, -0.0833,       0}, \
		{-0.0833,  -0.166, -0.0833}, \
		{      0, -0.0833,       0}, \
	}, \
	{ \
		{-0.0833, -0.166, -0.0833}, \
		{ -0.166,  2.666,  -0.166}, \
		{-0.0833, -0.166, -0.0833}, \
	}, \
	{ \
		{      0, -0.0833,       0}, \
		{-0.0833,  -0.166, -0.0833}, \
		{      0, -0.0833,       0}, \
	}, \
}
template <>
struct KernelInfo<KernelType::j3d27pt>{
	static constexpr int RAD = 1;
	static constexpr int FP = 54;
};
#define W_j3d27pt { \
	{ \
		{0.001, 0.002, 0.003}, \
		{0.004, 0.005, 0.006}, \
		{0.007, 0.008, 0.009}, \
	}, \
	{ \
		{0.010, 0.011, 0.012}, \
		{0.013, 0.649, 0.014}, \
		{0.015, 0.016, 0.017}, \
	}, \
	{ \
		{0.018, 0.019, 0.020}, \
		{0.021, 0.022, 0.023}, \
		{0.024, 0.025, 0.026}, \
	}, \
}

namespace utils{
#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOSTDEV
#define FORCEINLINE inline __attribute__ ((always_inline))
#endif
	int getCudaArch_host();
	constexpr int getCudaArch_device(){
#if __CUDA_ARCH__ >= 900
		return 900;
#else
		return 800;
#endif
	}
	HOSTDEV constexpr int align_to_pow2(int n){
		int m = 1;
		while (m < n) m <<= 1;
		return m;
	}
	template<KernelType KERNEL_T, typename VALUE_T>
	HOSTDEV constexpr VALUE_T kernel_w(int z, int y, int x){
		constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
		if constexpr (KERNEL_T == KernelType::j3d7pt){
			constexpr VALUE_T w[][RAD * 2 + 1][RAD * 2 + 1] = W_j3d7pt;
			return w[z][y][x];
		} else if constexpr (KERNEL_T == KernelType::j3d13pt){
			constexpr VALUE_T w[][RAD * 2 + 1][RAD * 2 + 1] = W_j3d13pt;
			return w[z][y][x];
		} else if constexpr (KERNEL_T == KernelType::poisson){
			constexpr VALUE_T w[][RAD * 2 + 1][RAD * 2 + 1] = W_poisson;
			return w[z][y][x];
		} else{
			constexpr VALUE_T w[][RAD * 2 + 1][RAD * 2 + 1] = W_j3d27pt;
			return w[z][y][x];
		}
	}
	template<KernelType KERNEL_T, typename VALUE_T, int kernel_z = 0, int kernel_y = 0, int kernel_x = 0>
	HOSTDEV FORCEINLINE void kernelCompute(VALUE_T *reg_output, VALUE_T (*reg_input)[KernelInfo<KERNEL_T>::RAD * 2 + 1][KernelInfo<KERNEL_T>::RAD * 2 + 1]){
		constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
		if constexpr (kernel_w<KERNEL_T, VALUE_T>(kernel_z, kernel_y, kernel_x) != 0)
			*reg_output += kernel_w<KERNEL_T, VALUE_T>(kernel_z, kernel_y, kernel_x) * reg_input[kernel_z][kernel_y][kernel_x];
		if constexpr (kernel_x + 1 < RAD * 2 + 1){
			kernelCompute<KERNEL_T, VALUE_T, kernel_z, kernel_y, kernel_x + 1>(reg_output, reg_input);
		} else if constexpr (kernel_y + 1 < RAD * 2 + 1){
			kernelCompute<KERNEL_T, VALUE_T, kernel_z, kernel_y + 1, 0>(reg_output, reg_input);
		} else if constexpr (kernel_z + 1 < RAD * 2 + 1){
			kernelCompute<KERNEL_T, VALUE_T, kernel_z + 1, 0, 0>(reg_output, reg_input);
		}
	}
#undef HOSTDEV
#undef FORCEINLINE
}

template<KernelType KERNEL_T, typename VALUE_T>
void SpatialTiling(VALUE_T *init_data, VALUE_T *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j3d7pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j3d13pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::poisson, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j3d27pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);

template<KernelType KERNEL_T, typename VALUE_T>
void FastStencil(VALUE_T *init_data, VALUE_T *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j3d7pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j3d13pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::poisson, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j3d27pt, double>(double *init_data, double *output_data, int NZ, int NY, int NX, int T, bool enable_warmup);
