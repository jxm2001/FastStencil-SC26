#pragma once
#include "defines.h"
#include "logger.hpp"

namespace FastStencilParams{
	template<KernelType KERNEL_T, typename VALUE_T>
	struct DefaultParams;
	template<KernelType KERNEL_T, typename VALUE_T, int CUDA_ARCH>
	struct ArchSpecificParams : public DefaultParams<KERNEL_T, VALUE_T> {};
	template<KernelType KERNEL_T, typename VALUE_T>
	struct Params : public ArchSpecificParams<KERNEL_T, VALUE_T, utils::getCudaArch_device()> {};

#ifdef DISABLE_TEMPORAL_TILING
	constexpr const char* impl_name = "FastStencil (No Temporal)";
	template <>
	struct DefaultParams<KernelType::j2d5pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 27;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j2d9pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 27;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j2ds9pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 19;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 4;
	};
	template <>
	struct DefaultParams<KernelType::j2d25pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 17;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 4;
	};
	template <>
	struct DefaultParams<KernelType::j2d13pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 15;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 6;
	};
	template <>
	struct DefaultParams<KernelType::j2d49pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 15;
		static constexpr int warps_num = 1;
		static constexpr int UNROLL_COUNT = 2;
	};
#else
	constexpr const char* impl_name = "FastStencil";
	template <>
	struct DefaultParams<KernelType::j2d5pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 34;
		static constexpr int warps_num = 12;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j2d9pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 23;
		static constexpr int warps_num = 12;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j2ds9pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 17;
		static constexpr int warps_num = 12;
		static constexpr int UNROLL_COUNT = 4;
	};
	template <>
	struct DefaultParams<KernelType::j2d25pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 23;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 4;
	};
	template <>
	struct DefaultParams<KernelType::j2d13pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 19;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 6;
	};
	template <>
	struct DefaultParams<KernelType::j2d49pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 17;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 6;
	};

	template <>
	struct ArchSpecificParams<KernelType::j2d5pt, double, 900>{
		static constexpr int core_t_tile = 4;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 13;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct ArchSpecificParams<KernelType::j2d9pt, double, 900>{
		static constexpr int core_t_tile = 3;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 15;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 3;
	};
	template <>
	struct ArchSpecificParams<KernelType::j2ds9pt, double, 900>{
		static constexpr int core_t_tile = 2;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 14;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 4;
	};
	template <>
	struct ArchSpecificParams<KernelType::j2d49pt, double, 900>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_y_tile_stream = 1;
		static constexpr int reg_x_tile = 15;
		static constexpr int warps_num = 8;
		static constexpr int UNROLL_COUNT = 6;
	};
#endif

	template<KernelType KERNEL_T, typename VALUE_T>
	class DeviceParams {
	public:
		static DeviceParams& getInstance() {
			static DeviceParams instance;
			return instance;
		}
		DeviceParams(const DeviceParams&) = delete;
		DeviceParams& operator=(const DeviceParams&) = delete;
		int get_CTX_SZ(){ return CTX_SZ; }
		int get_BT(){ return BT; }
		int get_gm_x_halo_tile(){ return gm_x_halo_tile; }
		int get_gm_x_vaild_tile(){ return gm_x_vaild_tile; }
		int get_sm_y_tile_mod(){ return sm_y_tile_mod; }

	private:
		int CTX_SZ;
		int BT;
		int gm_x_halo_tile;
		int gm_x_vaild_tile;
		int sm_y_tile_mod;
		template<int CUDA_ARCH>
		void init(){
			constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
			constexpr int UNROLL_COUNT = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::UNROLL_COUNT;
			constexpr int core_t_tile = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::core_t_tile;
			constexpr int core_y_tile_stream = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::core_y_tile_stream;
			constexpr int reg_x_tile = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::reg_x_tile;
			constexpr int warps_num = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::warps_num;
			constexpr int CTX_SZ = warps_num * 32;
			constexpr int BT = warps_num * core_t_tile;
			constexpr int gm_x_halo_tile = reg_x_tile * 32;
			constexpr int gm_x_vaild_tile = gm_x_halo_tile - RAD * BT * 2;
			constexpr int sm_y_tile = core_y_tile_stream * (warps_num + 2);
			constexpr int sm_y_tile_mod = utils::align_to_pow2(sm_y_tile);
			this->CTX_SZ = CTX_SZ;
			this->BT = BT;
			this->gm_x_halo_tile = gm_x_halo_tile;
			this->gm_x_vaild_tile = gm_x_vaild_tile;
			this->sm_y_tile_mod = sm_y_tile_mod;
		}

		DeviceParams() {
			int CUDA_ARCH = utils::getCudaArch_host();
			if(CUDA_ARCH >= 900)
				init<900>();
			else
				init<800>();
		}
	};

	template<KernelType KERNEL_T, typename VALUE_T>
	void getDefaultProblemSize(int &NY, int &NX, int &T, bool &checker){
		const int BT = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_BT();
		const int gm_x_vaild_tile = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_gm_x_vaild_tile();
		T = BT;
		if(utils::getCudaArch_host() == 900)
			NX = gm_x_vaild_tile * 33;
		else if(utils::getCudaArch_host() == 800)
			NX = gm_x_vaild_tile * 12;
		else{
			LOG_ERROR("Incorrect number of arguments. Usage: ./<perf_exec> <kernel_type> <NY> <NX> <T> <checker>");
			exit(1);
		}
		NY = 10000;
		checker = false;
	}
}
