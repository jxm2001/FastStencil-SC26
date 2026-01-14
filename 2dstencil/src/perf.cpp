#include <random>
#include <cstring>
#include <omp.h>
#include "defines.h"
#include "logger.hpp"
#include "FastStencil/params.cuh"
#include "SpatialTiling/params.cuh"

#ifndef PERF_FUNC
#define PERF_FUNC FastStencil
#endif

#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
using namespace CONCAT(PERF_FUNC, Params);

KernelType utils::arg2kernelType(char *str){
	if(strcmp(str, "j2d5pt") == 0){
		return KernelType::j2d5pt;
	}
	else if(strcmp(str, "j2d9pt") == 0){
		return KernelType::j2d9pt;
	}
	else if(strcmp(str, "j2ds9pt") == 0){
		return KernelType::j2ds9pt;
	}
	else if(strcmp(str, "j2d25pt") == 0){
		return KernelType::j2d25pt;
	}
	else if(strcmp(str, "j2d13pt") == 0){
		return KernelType::j2d13pt;
	}
	else if(strcmp(str, "j2d49pt") == 0){
		return KernelType::j2d49pt;
	}
	else{
		return KernelType::unknow;
	}
}

template<KernelType KERNEL_T, typename VALUE_T>
void cpuKernel(VALUE_T *init_data, VALUE_T *vaild_data, int NY, int NX, int T){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	size_t data_memsize = sizeof(VALUE_T)*NY*NX;
	VALUE_T *hm_input = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *hm_output = (VALUE_T *)aligned_alloc(64, data_memsize);
	memcpy(hm_input, init_data, data_memsize);
	for(int t = 0; t < T; t++){
		if(t != 0)
			std::swap(hm_input, hm_output);
		#pragma omp parallel
		for (int y = 0; y < NY; y++){
			for (int x = 0; x < NX; x++){
				VALUE_T reg_input[RAD * 2 + 1][RAD * 2 + 1], reg_output = 0;
				#pragma unroll
				for (int data_y = 0; data_y < RAD * 2 + 1; data_y++){
					int hm_y = std::min(std::max(y - RAD + data_y, 0), NY - 1);
					#pragma unroll
					for (int data_x = 0; data_x < RAD * 2 + 1; data_x++){
						int hm_x = std::min(std::max(x - RAD + data_x, 0), NX - 1);
						reg_input[data_y][data_x] = hm_input[Offset_gm(hm_y, hm_x)];
					}
				}
				utils::kernelCompute<KERNEL_T, VALUE_T>(&reg_output, reg_input);
				hm_output[Offset_gm(y, x)] = reg_output;
			}
		}
	}
	memcpy(vaild_data, hm_output, data_memsize);
	free(hm_input);
	free(hm_output);
}

template<KernelType KERNEL_T, typename VALUE_T>
void cpuCheck(VALUE_T *output_data, VALUE_T *vaild_data, int NY, int NX, int T){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	constexpr double eps = 1e-6;
	int error = 0, halo = T * RAD;
	for (int y = halo; y < NY - halo; y++) {
		int error_x = 0;
		for (int x = halo; x < NX - halo; x++) {
			VALUE_T reg_output = output_data[Offset_gm(y, x)];
			VALUE_T reg_vaild = vaild_data[Offset_gm(y, x)];
			double diff = std::abs((reg_output - reg_vaild) / reg_vaild);
			if(std::isnan(reg_output) || diff > eps){
				error_x++;
				if(error_x > 4)
					break;
				LOG_DEBUG("correct[%d][%d] = %f, wrong = %f, diff %f: FAILED!", y, x, reg_vaild, reg_output, diff);
			}
		}
		error += error_x > 0;
		if(error > 4)
			break;
	}
	if(error == 0)
		LOG_DEBUG("pass");
	else
		LOG_DEBUG("failed");
}

template<KernelType KERNEL_T, typename VALUE_T>
void perf(int NY, int NX, int T, bool checker){
	size_t data_memsize = sizeof(VALUE_T)*NY*NX;

	VALUE_T *init_data = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *output_data = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *vaild_data = (VALUE_T *)aligned_alloc(64, data_memsize);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		std::mt19937 gen(tid);
		std::uniform_int_distribution<int> dis(0, 1024);
		#pragma omp for schedule(static)
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				init_data[Offset_gm(y,x)] = dis(gen);
			}
		}
	}


	if(checker)
		cpuKernel<KERNEL_T, VALUE_T>(init_data, vaild_data, NY, NX, T);

	PERF_FUNC<KERNEL_T, VALUE_T>(init_data, output_data, NY, NX, T, true);

	if(checker)
		cpuCheck<KERNEL_T, VALUE_T>(output_data, vaild_data, NY, NX, T);

	free(init_data);
	free(output_data);
	free(vaild_data);
}

int main(int argc, char * argv[]) {
	if(argc != 2 && argc != 6){
		LOG_ERROR("Incorrect number of arguments. Usage: ./<perf_exec> <kernel_type> <NY> <NX> <T> <checker>");
		exit(1);
	}

	KernelType KERNEL_T = utils::arg2kernelType(argv[1]);
	if(KERNEL_T == KernelType::unknow){
		LOG_ERROR("Unknown kernel type: %s. Supported types: j2d5pt, j2d9pt, j2ds9pt, j2d25pt, j2d49pt", argv[1]);
		exit(1);
	}

	int NY, NX, T;
	bool checker;
	if(argc == 6){
		NY = atoi(argv[2]);
		NX = atoi(argv[3]);
		T = atoi(argv[4]);
		checker = atoi(argv[5]);
	}
	else{
		if(KERNEL_T == KernelType::j2d5pt)
			getDefaultProblemSize<KernelType::j2d5pt, double>(NY, NX, T, checker);
		else if(KERNEL_T == KernelType::j2d9pt)
			getDefaultProblemSize<KernelType::j2d9pt, double>(NY, NX, T, checker);
		else if(KERNEL_T == KernelType::j2ds9pt)
			getDefaultProblemSize<KernelType::j2ds9pt, double>(NY, NX, T, checker);
		else if(KERNEL_T == KernelType::j2d25pt)
			getDefaultProblemSize<KernelType::j2d25pt, double>(NY, NX, T, checker);
		else if(KERNEL_T == KernelType::j2d13pt)
			getDefaultProblemSize<KernelType::j2d13pt, double>(NY, NX, T, checker);
		else
			getDefaultProblemSize<KernelType::j2d49pt, double>(NY, NX, T, checker);
	}

	LOG_INFO("NY %d, NX %d, T %d, checker %d", NY, NX, T, checker);

	if(KERNEL_T == KernelType::j2d5pt)
		perf<KernelType::j2d5pt, double>(NY, NX, T, checker);
	else if(KERNEL_T == KernelType::j2d9pt)
		perf<KernelType::j2d9pt, double>(NY, NX, T, checker);
	else if(KERNEL_T == KernelType::j2ds9pt)
		perf<KernelType::j2ds9pt, double>(NY, NX, T, checker);
	else if(KERNEL_T == KernelType::j2d25pt)
		perf<KernelType::j2d25pt, double>(NY, NX, T, checker);
	else if(KERNEL_T == KernelType::j2d13pt)
		perf<KernelType::j2d13pt, double>(NY, NX, T, checker);
	else
		perf<KernelType::j2d49pt, double>(NY, NX, T, checker);

	return 0;
}
