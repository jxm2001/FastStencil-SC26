#pragma once
#define Offset_gm(i,j) ((j) + NX * (i))

enum class KernelType{
	j2d5pt,
	j2d9pt,
	j2ds9pt,
	j2d25pt,
	j2d13pt,
	j2d49pt,
	unknow
};

template<KernelType KERNEL_T>
struct KernelInfo;

template <>
struct KernelInfo<KernelType::j2d5pt>{
	static constexpr int RAD = 1;
	static constexpr int FP = 10;
};
#define W_j2d5pt { \
	{   0, 0.01,    0}, \
	{0.02,  0.9, 0.03}, \
	{   0, 0.04,    0}, \
}
template <>
struct KernelInfo<KernelType::j2d9pt>{
	static constexpr int RAD = 1;
	static constexpr int FP = 18;
};
#define W_j2d9pt { \
	{0.01, 0.02, 0.03}, \
	{0.04, 0.64, 0.05}, \
	{0.06, 0.07, 0.08}, \
}
template <>
struct KernelInfo<KernelType::j2ds9pt>{
	static constexpr int RAD = 2;
	static constexpr int FP = 18;
};
#define W_j2ds9pt { \
	{   0,    0, 0.01,    0,    0}, \
	{   0,    0, 0.02,    0,    0}, \
	{0.03, 0.04, 0.64, 0.05, 0.06}, \
	{   0,    0, 0.07,    0,    0}, \
	{   0,    0, 0.08,    0,    0}, \
}
template <>
struct KernelInfo<KernelType::j2d25pt>{
	static constexpr int RAD = 2;
	static constexpr int FP = 50;
};
#define W_j2d25pt { \
	{0.001, 0.002, 0.003, 0.004, 0.005}, \
	{0.006, 0.007, 0.008, 0.009, 0.010}, \
	{0.011, 0.012, 0.700, 0.013, 0.014}, \
	{0.015, 0.016, 0.017, 0.018, 0.019}, \
	{0.020, 0.021, 0.022, 0.023, 0.024}, \
}
template <>
struct KernelInfo<KernelType::j2d13pt>{
	static constexpr int RAD = 3;
	static constexpr int FP = 26;
};
#define W_j2d13pt { \
	{   0,    0,    0, 0.01,    0,    0,    0}, \
	{   0,    0,    0, 0.02,    0,    0,    0}, \
	{   0,    0,    0, 0.03,    0,    0,    0}, \
	{0.04, 0.05, 0.06, 0.22, 0.07, 0.08, 0.09}, \
	{   0,    0,    0, 0.10,    0,    0,    0}, \
	{   0,    0,    0, 0.11,    0,    0,    0}, \
	{   0,    0,    0, 0.12,    0,    0,    0}, \
}
template <>
struct KernelInfo<KernelType::j2d49pt>{
	static constexpr int RAD = 3;
	static constexpr int FP = 98;
};
#define W_j2d49pt { \
	{0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007}, \
	{0.0008, 0.0009, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014}, \
	{0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.0020, 0.0021}, \
	{0.0022, 0.0023, 0.0024, 0.8824, 0.0025, 0.0026, 0.0027}, \
	{0.0028, 0.0029, 0.0030, 0.0031, 0.0032, 0.0033, 0.0034}, \
	{0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.0040, 0.0041}, \
	{0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048}, \
}

namespace utils{
#ifdef __CUDACC__
#define HOSTDEV __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOSTDEV
#define FORCEINLINE inline __attribute__ ((always_inline))
#endif
	KernelType arg2kernelType(char *str);
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
	HOSTDEV constexpr VALUE_T kernel_w(int y, int x){
		constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
		if constexpr (KERNEL_T == KernelType::j2d5pt){
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2d5pt;
			return w[y][x];
		} else if constexpr (KERNEL_T == KernelType::j2d9pt){
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2d9pt;
			return w[y][x];
		} else if constexpr (KERNEL_T == KernelType::j2ds9pt){
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2ds9pt;
			return w[y][x];
		} else if constexpr (KERNEL_T == KernelType::j2d25pt){
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2d25pt;
			return w[y][x];
		} else if constexpr (KERNEL_T == KernelType::j2d13pt){
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2d13pt;
			return w[y][x];
		} else{
			constexpr VALUE_T w[][RAD * 2 + 1] = W_j2d49pt;
			return w[y][x];
		}
	}
	template<KernelType KERNEL_T, typename VALUE_T, int kernel_y = 0, int kernel_x = 0>
	HOSTDEV FORCEINLINE void kernelCompute(VALUE_T *reg_output, VALUE_T (*reg_input)[KernelInfo<KERNEL_T>::RAD * 2 + 1]){
		constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
		if constexpr (kernel_w<KERNEL_T, VALUE_T>(kernel_y, kernel_x) != 0)
			*reg_output += kernel_w<KERNEL_T, VALUE_T>(kernel_y, kernel_x) * reg_input[kernel_y][kernel_x];
		if constexpr (kernel_x + 1 < RAD * 2 + 1){
			kernelCompute<KERNEL_T, VALUE_T, kernel_y, kernel_x + 1>(reg_output, reg_input);
		} else if constexpr (kernel_y + 1 < RAD * 2 + 1){
			kernelCompute<KERNEL_T, VALUE_T, kernel_y + 1, 0>(reg_output, reg_input);
		}
	}
#undef HOSTDEV
#undef FORCEINLINE
}

template<KernelType KERNEL_T, typename VALUE_T>
void SpatialTiling(VALUE_T *init_data, VALUE_T *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2d5pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2d9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2ds9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2d25pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2d13pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void SpatialTiling<KernelType::j2d49pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);

template<KernelType KERNEL_T, typename VALUE_T>
void FastStencil(VALUE_T *init_data, VALUE_T *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2d5pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2d9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2ds9pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2d25pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2d13pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
extern template void FastStencil<KernelType::j2d49pt, double>(double *init_data, double *output_data, int NY, int NX, int T, bool enable_warmup);
