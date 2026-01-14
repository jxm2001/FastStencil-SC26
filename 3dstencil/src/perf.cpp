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

KernelType arg2kernelType(char *str){
	if(strcmp(str, "j3d7pt") == 0){
		return KernelType::j3d7pt;
	}
	else if(strcmp(str, "j3d13pt") == 0){
		return KernelType::j3d13pt;
	}
	else if(strcmp(str, "poisson") == 0){
		return KernelType::poisson;
	}
	else if(strcmp(str, "j3d27pt") == 0){
		return KernelType::j3d27pt;
	}
	else{
		return KernelType::unknow;
	}
}

template<KernelType KERNEL_T, typename VALUE_T>
void cpuKernel(VALUE_T *init_data, VALUE_T *vaild_data, int NZ, int NY, int NX, int T){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	size_t data_memsize = sizeof(VALUE_T)*NZ*NY*NX;
	VALUE_T *hm_input = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *hm_output = (VALUE_T *)aligned_alloc(64, data_memsize);
	memcpy(hm_input, init_data, data_memsize);
	for(int t = 0; t < T; t++){
		if(t != 0)
			std::swap(hm_input, hm_output);
		#pragma omp parallel
		for (int z = 0; z < NZ; z++){
			for (int y = 0; y < NY; y++){
				for (int x = 0; x < NX; x++){
					VALUE_T reg_input[RAD * 2 + 1][RAD * 2 + 1][RAD * 2 + 1], reg_output = 0;
					#pragma unroll
					for (int data_z = 0; data_z < RAD * 2 + 1; data_z++){
						int hm_z = std::min(std::max(z - RAD + data_z, 0), NZ - 1);
						#pragma unroll
						for (int data_y = 0; data_y < RAD * 2 + 1; data_y++){
							int hm_y = std::min(std::max(y - RAD + data_y, 0), NY - 1);
							#pragma unroll
							for (int data_x = 0; data_x < RAD * 2 + 1; data_x++){
								int hm_x = std::min(std::max(x - RAD + data_x, 0), NX - 1);
								reg_input[data_z][data_y][data_x] = hm_input[Offset_gm(hm_z, hm_y, hm_x)];
							}
						}
					}
					utils::kernelCompute<KERNEL_T, VALUE_T>(&reg_output, reg_input);
					hm_output[Offset_gm(z, y, x)] = reg_output;
				}
			}
		}
	}
	memcpy(vaild_data, hm_output, data_memsize);
	free(hm_input);
	free(hm_output);
}

template<KernelType KERNEL_T, typename VALUE_T>
void cpuCheck(VALUE_T *output_data, VALUE_T *vaild_data, int NZ, int NY, int NX, int T){
	constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
	constexpr double eps = 1e-6;
	int error = 0, halo = T * RAD;
	for (int z = halo; z < NZ - halo; z++) {
		int error_y = 0;
		for (int y = halo; y < NY - halo; y++) {
			int error_x = 0;
			for (int x = halo; x < NX - halo; x++) {
				VALUE_T reg_output = output_data[Offset_gm(z, y, x)];
				VALUE_T reg_vaild = vaild_data[Offset_gm(z, y, x)];
				double diff = std::abs((reg_output - reg_vaild) / reg_vaild);
				if(std::isnan(reg_output) || diff > eps){
					error_x++;
					if(error_x > 4)
						break;
					LOG_DEBUG("correct[%d][%d][%d] = %f, wrong = %f, diff %f: FAILED!", z, y, x, reg_vaild, reg_output, diff);
				}
			}
			error_y += error_x > 0;
			if(error_y > 4)
				break;
		}
		error += error_y > 0;
		if(error > 4)
			break;
	}
	if(error == 0)
		LOG_DEBUG("pass");
	else
		LOG_DEBUG("failed");
}

template<KernelType KERNEL_T, typename VALUE_T>
void perf(int NZ, int NY, int NX, int T, bool checker){
	size_t data_memsize = sizeof(VALUE_T)*NZ*NY*NX;

	VALUE_T *init_data = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *output_data = (VALUE_T *)aligned_alloc(64, data_memsize);
	VALUE_T *vaild_data = (VALUE_T *)aligned_alloc(64, data_memsize);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		std::mt19937 gen(tid);
		std::uniform_int_distribution<int> dis(0, 1024);
		#pragma omp for schedule(static)
		for (int z = 0; z < NZ; z++) {
			for (int y = 0; y < NY; y++) {
				for (int x = 0; x < NX; x++) {
					init_data[Offset_gm(z, y, x)] = dis(gen);
				}
			}
		}
	}


	if(checker)
		cpuKernel<KERNEL_T, VALUE_T>(init_data, vaild_data, NZ, NY, NX, T);

	PERF_FUNC<KERNEL_T, VALUE_T>(init_data, output_data, NZ, NY, NX, T, true);
	if(checker)
		cpuCheck<KERNEL_T, VALUE_T>(output_data, vaild_data, NZ, NY, NX, T);

	free(init_data);
	free(output_data);
	free(vaild_data);
}

int main(int argc, char * argv[]) {
	if(argc != 2 && argc != 7){
		LOG_ERROR("Incorrect number of arguments. Usage: ./<perf_exec> <kernel_type> <NZ> <NY> <NX> <T> <checker>");
		exit(1);
	}

	KernelType KERNEL_T = arg2kernelType(argv[1]);
	if(KERNEL_T == KernelType::unknow){
		LOG_ERROR("Unknown kernel type: %s. Supported types: j3d7pt, j3d13pt, poisson, j3d27pt", argv[1]);
		exit(1);
	}

	int NZ, NY, NX, T;
	bool checker;
	if(argc == 7){
		NZ = atoi(argv[2]);
		NY = atoi(argv[3]);
		NX = atoi(argv[4]);
		T = atoi(argv[5]);
		checker = atoi(argv[6]);
	}
	else{
		if(KERNEL_T == KernelType::j3d7pt)
			getDefaultProblemSize<KernelType::j3d7pt, double>(NZ, NY, NX, T, checker);
		else if(KERNEL_T == KernelType::j3d13pt)
			getDefaultProblemSize<KernelType::j3d13pt, double>(NZ, NY, NX, T, checker);
		else if(KERNEL_T == KernelType::poisson)
			getDefaultProblemSize<KernelType::poisson, double>(NZ, NY, NX, T, checker);
		else
			getDefaultProblemSize<KernelType::j3d27pt, double>(NZ, NY, NX, T, checker);
	}

	LOG_INFO("Info: NZ %d, NY %d, NX %d, T %d, checker %d", NZ, NY, NX, T, checker);

	if(KERNEL_T == KernelType::j3d7pt)
		perf<KernelType::j3d7pt, double>(NZ, NY, NX, T, checker);
	else if(KERNEL_T == KernelType::j3d13pt)
		perf<KernelType::j3d13pt, double>(NZ, NY, NX, T, checker);
	else if(KERNEL_T == KernelType::poisson)
		perf<KernelType::poisson, double>(NZ, NY, NX, T, checker);
	else
		perf<KernelType::j3d27pt, double>(NZ, NY, NX, T, checker);

	return 0;
}
