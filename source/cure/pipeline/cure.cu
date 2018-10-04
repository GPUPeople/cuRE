


#include "pipeline_module.cuh"

#include "primitive_types.cuh"
#include "../shaders/simple_shading.cuh"
#include "../shaders/clipspace.cuh"
#include "../shaders/eyecandy.cuh"
#include "../shaders/blending.cuh"
#include "../shaders/ocean.cuh"
#include "../shaders/checkerboard.cuh"
#include "../shaders/blend_shader.cuh"


using SimpleVertexBuffer = VertexBuffer<32U>;

struct AnimatedVertexBuffer : SimpleVertexBuffer
{
	__device__
	static const float4* attributes(unsigned int index)
	{
		return SimpleVertexBuffer::attributes(index);
	}
};

using SimpleVertexAttributes = InputVertexAttributes<
	VertexBufferAttributes<SimpleVertexBuffer, math::float3, math::float3>
>;
INSTANTIATE_PIPELINE(simple_shading_tris, true, SimpleVertexAttributes, IndexedTriangleList, SimpleVertexShader, CoverageShader, SimpleFragmentShader, NoBlending);
INSTANTIATE_PIPELINE(simple_shading_quads, true, SimpleVertexAttributes, IndexedQuadList, SimpleVertexShader, CoverageShader, SimpleFragmentShader, NoBlending);

using TexturedVertexAttributes = InputVertexAttributes<
	VertexBufferAttributes<SimpleVertexBuffer, math::float3, math::float3, math::float2>
>;
INSTANTIATE_PIPELINE(textured_shading_tris, true, TexturedVertexAttributes, IndexedTriangleList, TexturedVertexShader, CoverageShader, TexturedFragmentShader<false>, TextureBlending);
INSTANTIATE_PIPELINE(textured_shading_quads, true, TexturedVertexAttributes, IndexedQuadList, TexturedVertexShader, CoverageShader, TexturedFragmentShader<false>, TextureBlending);

INSTANTIATE_PIPELINE(vertex_heavy_tris, true, SimpleVertexAttributes, IndexedTriangleList, VertexHeavyVertexShader, CoverageShader, VertexHeavyFragmentShader, NoBlending);
INSTANTIATE_PIPELINE(fragment_heavy_tris, true, SimpleVertexAttributes, IndexedTriangleList, FragmentHeavyVertexShader, CoverageShader, FragmentHeavyFragmentShader, NoBlending);


using WaterVertexBuffer = VertexBuffer<16U>;
using WaterVertexAttributes = InputVertexAttributes<VertexBufferAttributes<WaterVertexBuffer, math::float4>>;
INSTANTIATE_PIPELINE(ocean_adaptive, ENABLE_OCEAN_DEMO, WaterVertexAttributes, IndexedAdaptiveQuadList<WaveQuadTriangulationShader>, WaterVertexShader, CoverageShader, WaterFragmentShader<false>, WaterBlending);
INSTANTIATE_PIPELINE(ocean_normal, ENABLE_OCEAN_DEMO, WaterVertexAttributes, IndexedQuadList, WaterVertexShader, CoverageShader, WaterFragmentShader<false>, WaterBlending);
INSTANTIATE_PIPELINE(ocean_adaptive_wire, ENABLE_OCEAN_DEMO, WaterVertexAttributes, IndexedAdaptiveQuadList<WaveQuadTriangulationShader>, WaterVertexShader, CoverageShader, WaterFragmentShader<true>, WaterBlending);
INSTANTIATE_PIPELINE(ocean_normal_wire, ENABLE_OCEAN_DEMO, WaterVertexAttributes, IndexedQuadList, WaterVertexShader, CoverageShader, WaterFragmentShader<true>, WaterBlending);

using ClipspaceVertexBuffer = VertexBuffer<16U>;
using ClipspaceVertexAttributes = InputVertexAttributes<
	VertexBufferAttributes<ClipspaceVertexBuffer, math::float4>
>;
INSTANTIATE_PIPELINE(clipspace_shading, true, ClipspaceVertexAttributes, IndexedTriangleList, ClipspaceVertexShader, CoverageShader, ClipspaceFragmentShader, ClipspaceBlending);
INSTANTIATE_PIPELINE(vertex_heavy_clipspace_shading, true, ClipspaceVertexAttributes, IndexedTriangleList, VertexHeavyClipspaceVertexShader, CoverageShader, VertexHeavyClipspaceFragmentShader, ClipspaceBlending);
INSTANTIATE_PIPELINE(fragment_heavy_clipspace_shading, true, ClipspaceVertexAttributes, IndexedTriangleList, ClipspaceVertexShader, CoverageShader, FragmentHeavyClipspaceFragmentShader, ClipspaceBlending);

using EyeCandyVertexBuffer = VertexBuffer<48U>;
using EyeCandyeVertexAttributes = InputVertexAttributes<VertexBufferAttributes<EyeCandyVertexBuffer, math::float4, math::float4, math::float4>>;
INSTANTIATE_PIPELINE(eyecandy_shading, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CoverageShader, EyeCandyFragmentShader<false>, EyeCandyBlending);
INSTANTIATE_PIPELINE(vertex_heavy_eyecandy_shading, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShaderVertexHeavy, CoverageShader, EyeCandyFragmentShaderVertexHeavy, EyeCandyBlending);
INSTANTIATE_PIPELINE(fragment_heavy_eyecandy_shading, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CoverageShader, EyeCandyFragmentShaderFragmentHeavy, EyeCandyBlending);


using BlendVertexBuffer = VertexBuffer<32U>;
using BlendVertexAttributes = InputVertexAttributes<VertexBufferAttributes<BlendVertexBuffer, math::float2, math::float3, math::float3>>;
INSTANTIATE_PIPELINE(blend_demo, true, BlendVertexAttributes, IndexedTriangleList, BlendVertexShader, CoverageShader, BlendFragmentShader, BlendBlending);

using IsoBlendVertexBuffer = VertexBuffer<48U>;
using IsoBlendVertexAttributes = InputVertexAttributes<VertexBufferAttributes<IsoBlendVertexBuffer, math::float3, math::float3, math::float4>>;
INSTANTIATE_PIPELINE(iso_blend_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, IsoBlendVertexShader, CoverageShader, IsoBlendFragmentShader, IsoBlendBlending);

INSTANTIATE_PIPELINE(iso_stipple_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, IsoBlendVertexShader, CoverageShader, IsoBlendFragmentShader, NoBlending);

INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Normal>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<FiftyFifty>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Multiply>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Screen>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Darken>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Lighten>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<ColorDodge>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<ColorBurn>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<HardLight>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<SoftLight>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Overlay>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Difference>);
//INSTANTIATE_PIPELINE(glyph_demo, true, IsoBlendVertexAttributes, IndexedTriangleList, GlyphVertexShader, CoverageShader, GlyphFragmentShader, SeparableBlendOp<Exclusion>);

INSTANTIATE_PIPELINE(checkerboard_demo, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CheckerboardCoverageShader, EyeCandyFragmentShader<false>, EyeCandyBlending);
INSTANTIATE_PIPELINE(checkerboard_quad_demo, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CheckerboardQuadCoverageShader, EyeCandyFragmentShader<false>, EyeCandyBlending);
INSTANTIATE_PIPELINE(checkerboard_fragment_demo, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CoverageShader, EyeCandyCoverageFragmentShader, EyeCandyBlending);
INSTANTIATE_PIPELINE(checkerboard_quad_fragment_demo, true, EyeCandyeVertexAttributes, IndexedTriangleList, EyeCandyVertexShader, CoverageShader, EyeCandyQuadCoverageFragmentShader, EyeCandyBlending);
