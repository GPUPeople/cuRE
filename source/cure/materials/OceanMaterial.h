


#ifndef INCLUDED_CURE_MATERIAL_OCEAN
#define INCLUDED_CURE_MATERIAL_OCEAN

#include <cstdint>

#include <CUDA/array.h>

#include <Resource.h>


namespace cuRE
{
	class Pipeline;

	class OceanMaterial : public Material
	{
		OceanMaterial(const OceanMaterial&) = delete;
		OceanMaterial& operator =(const OceanMaterial&) = delete;

		CU::unique_array color_array;
		CU::unique_mipmapped_array normal_array;

		Pipeline& pipeline;

	public:
		OceanMaterial(Pipeline& pipeline, const float* img_data, size_t width, size_t height, const std::uint32_t* normal_data, size_t n_width, size_t n_height, unsigned int n_levels);

		void draw(const ::Geometry* geometry) const override;
		void draw(const ::Geometry* geometry, int start, int num_indices) const override;
	};
}

#endif  // INCLUDED_CURE_MATERIAL_OCEAN
