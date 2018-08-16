//
//
//
//#ifndef INCLUDED_CURE_MATERIAL_TEXTURED
//#define INCLUDED_CURE_MATERIAL_TEXTURED
//
//#include "Material.h"
//#include <CUDA/module.h>
//
//namespace CURE
//{
//	class Connector;
//	class TexturedMaterial : public CURE::Material
//	{
//		TexturedMaterial(const TexturedMaterial&) = delete;
//		TexturedMaterial& operator =(const TexturedMaterial&) = delete;
//
//		Connector* connector;
//
//	public:
//		TexturedMaterial(const math::float4& color, CU::unique_module& module, Connector* connector);
//
//		void draw(const ::Geometry* geometry) const;
//		void setModel(const math::float3x4& mat);
//		void setCamera(const math::float4x4& PV, const math::float3& pos);
//		void remove()
//		{
//			connector = nullptr;
//		}
//	};
//}
//
//#endif  // INCLUDED_CURE_MATERIAL_TEXTURED
