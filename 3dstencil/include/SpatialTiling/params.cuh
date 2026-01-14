#pragma once
#include "defines.h"

namespace SpatialTilingParams{
	struct DefaultParams{
		static constexpr int threads_z_num = 4;
		static constexpr int threads_y_num = 4;
		static constexpr int threads_x_num = 64;
	};
	template<KernelType KERNEL_T, typename VALUE_T>
	void getDefaultProblemSize(int &NZ, int &NY, int &NX, int &T, bool &checker){
		T = 1;
		NZ = DefaultParams::threads_y_num * 400;
		NY = DefaultParams::threads_y_num * 100;
		NX = DefaultParams::threads_x_num * 4;
		checker = false;
	}
}
