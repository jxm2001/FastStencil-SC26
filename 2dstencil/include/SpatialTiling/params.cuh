#pragma once
#include "defines.h"

namespace SpatialTilingParams{
	struct DefaultParams{
		static constexpr int threads_y_num = 16;
		static constexpr int threads_x_num = 64;
	};
	template<KernelType KERNEL_T, typename VALUE_T>
	void getDefaultProblemSize(int &NY, int &NX, int &T, bool &checker){
		T = 1;
		NY = DefaultParams::threads_y_num * 400;
		NX = DefaultParams::threads_x_num * 200;
		checker = false;
	}
}
