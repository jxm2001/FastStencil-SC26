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
	struct DefaultParams<KernelType::j3d7pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 7;
		static constexpr int reg_x_tile = 5;
		static constexpr int warps_t_num = 1;
		static constexpr int warps_y_num = 8;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j3d13pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 5;
		static constexpr int reg_x_tile = 3;
		static constexpr int warps_t_num = 1;
		static constexpr int warps_y_num = 8;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::poisson, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 6;
		static constexpr int reg_x_tile = 5;
		static constexpr int warps_t_num = 1;
		static constexpr int warps_y_num = 8;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j3d27pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 8;
		static constexpr int reg_x_tile = 3;
		static constexpr int warps_t_num = 1;
		static constexpr int warps_y_num = 8;
		static constexpr int UNROLL_COUNT = 2;
	};
#else
	constexpr const char* impl_name = "FastStencil";
	template <>
	struct DefaultParams<KernelType::j3d7pt, double>{
		static constexpr int core_t_tile = 2;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 9;
		static constexpr int reg_x_tile = 2;
		static constexpr int warps_t_num = 2;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 6;
	};
	template <>
	struct DefaultParams<KernelType::j3d13pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 6;
		static constexpr int reg_x_tile = 3;
		static constexpr int warps_t_num = 2;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::poisson, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 8;
		static constexpr int reg_x_tile = 3;
		static constexpr int warps_t_num = 3;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct DefaultParams<KernelType::j3d27pt, double>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 8;
		static constexpr int reg_x_tile = 2;
		static constexpr int warps_t_num = 3;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 4;
	};

	template <>
	struct ArchSpecificParams<KernelType::j3d7pt, double, 900>{
		static constexpr int core_t_tile = 3;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 7;
		static constexpr int reg_x_tile = 2;
		static constexpr int warps_t_num = 2;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 2;
	};
	template <>
	struct ArchSpecificParams<KernelType::j3d27pt, double, 900>{
		static constexpr int core_t_tile = 1;
		static constexpr int core_z_tile_stream = 1;
		static constexpr int core_y_tile = 6;
		static constexpr int reg_x_tile = 3;
		static constexpr int warps_t_num = 3;
		static constexpr int warps_y_num = 4;
		static constexpr int UNROLL_COUNT = 2;
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
		int get_warps_t_num() { return warps_t_num; }
		int get_warps_y_num() { return warps_y_num; }
		int get_core_z_tile_stream() { return core_z_tile_stream; }
		int get_CTX_SZ(){ return CTX_SZ; }
		int get_BT(){ return BT; }
		int get_gm_x_halo_tile(){ return gm_x_halo_tile; }
		int get_gm_x_vaild_tile(){ return gm_x_vaild_tile; }
		int get_gm_y_halo_tile(){ return gm_y_halo_tile; }
		int get_gm_y_vaild_tile(){ return gm_y_vaild_tile; }
		int get_sm2_y_tile(){ return sm2_y_tile; }
		int get_sm_z_tile_mod(){ return sm_z_tile_mod; }

	private:
		int warps_t_num;
		int warps_y_num;
		int core_z_tile_stream;
		int CTX_SZ;
		int BT;
		int gm_x_halo_tile;
		int gm_x_vaild_tile;
		int gm_y_halo_tile;
		int gm_y_vaild_tile;
		int sm2_y_tile;
		int sm_z_tile_mod;
		template<int CUDA_ARCH>
		void init(){
			constexpr int RAD = KernelInfo<KERNEL_T>::RAD;
			constexpr int UNROLL_COUNT = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::UNROLL_COUNT;
			constexpr int core_t_tile = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::core_t_tile;
			constexpr int core_z_tile_stream = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::core_z_tile_stream;
			constexpr int core_y_tile = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::core_y_tile;
			constexpr int reg_x_tile = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::reg_x_tile;
			constexpr int warps_t_num = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::warps_t_num;
			constexpr int warps_y_num = ArchSpecificParams<KERNEL_T, VALUE_T, CUDA_ARCH>::warps_y_num;
			constexpr int warps_num = warps_t_num * warps_y_num;
			constexpr int CTX_SZ = warps_num * 32;
			constexpr int BT = warps_t_num * core_t_tile;
			constexpr int sm_z_tile = core_z_tile_stream * (warps_t_num + 1);
			constexpr int sm_z_tile_mod = utils::align_to_pow2(sm_z_tile);
			constexpr int gm_y_halo_tile = core_y_tile * warps_y_num + RAD * 2;
			constexpr int gm_y_vaild_tile = gm_y_halo_tile - RAD * BT * 2;
			constexpr int sm2_y_tile = (warps_y_num + 1) * RAD * 2;
			constexpr int gm_x_halo_tile = reg_x_tile * 32;
			constexpr int gm_x_vaild_tile = gm_x_halo_tile - RAD * BT * 2;
			if(core_t_tile > 1 && RAD * 2 > core_y_tile){
				LOG_ERROR("Illegal tiling parameters");
				exit(1);
			}
			this->warps_t_num = warps_t_num;
			this->warps_y_num = warps_y_num;
			this->core_z_tile_stream = core_z_tile_stream;
			this->CTX_SZ = CTX_SZ;
			this->BT = BT;
			this->gm_x_halo_tile = gm_x_halo_tile;
			this->gm_x_vaild_tile = gm_x_vaild_tile;
			this->gm_y_halo_tile = gm_y_halo_tile;
			this->gm_y_vaild_tile = gm_y_vaild_tile;
			this->sm2_y_tile = sm2_y_tile;
			this->sm_z_tile_mod = sm_z_tile_mod;
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
	void getDefaultProblemSize(int &NZ, int &NY, int &NX, int &T, bool &checker){
		const int BT = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_BT();
		const int gm_x_vaild_tile = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_gm_x_vaild_tile();
		const int gm_y_vaild_tile = DeviceParams<KERNEL_T, VALUE_T>::getInstance().get_gm_y_vaild_tile();
		T = BT;
		if(utils::getCudaArch_host() == 900){
			NX = gm_x_vaild_tile * 6;
			NY = gm_y_vaild_tile * 11;
		}
		else if(utils::getCudaArch_host() == 800){
			NX = gm_x_vaild_tile * 6;
			NY = gm_y_vaild_tile * 9;
		}
		else{
			LOG_ERROR("Incorrect number of arguments. Usage: ./<perf_exec> <kernel_type> <NZ> <NY> <NX> <T> <checker>");
			exit(0);
		}
		NZ = 2560;
		checker = false;
	}
}
