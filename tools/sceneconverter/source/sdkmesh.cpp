


#include <cctype>
#include <cstring>

#include <memory>
#include <string>

#include <algorithm>
#include <functional>
#include <numeric>

#include <vector>
#include <unordered_map>

#include <limits>

#include <cstdint>

//#include <utils/io.h>

#include <math/vector.h>

#include "sdkmesh.h"


namespace sdkmesh
{
	const unsigned int SDKMESH_FILE_VERSION = 101U;
	const unsigned int MAX_VERTEX_ELEMENTS = 32U;
	const unsigned int MAX_VERTEX_STREAMS = 16U;
	const unsigned int MAX_FRAME_NAME = 100U;
	const unsigned int MAX_MESH_NAME = 100U;
	const unsigned int MAX_SUBSET_NAME = 100U;
	const unsigned int MAX_MATERIAL_NAME = 100U;
	const unsigned int MAX_TEXTURE_NAME = 260U;
	const unsigned int MAX_MATERIAL_PATH = 260U;
	const unsigned int INVALID_FRAME = ~0U;
	const unsigned int INVALID_MESH = ~0U;
	const unsigned int INVALID_MATERIAL = ~0U;
	const unsigned int INVALID_SUBSET = ~0U;
	const unsigned int INVALID_ANIMATION_DATA = ~0U;
	const unsigned int INVALID_SAMPLER_SLOT = ~0U;
	const unsigned int ERROR_RESOURCE_VALUE = 1U;


	enum vertex_element_type
	{
		TYPE_FLOAT1		 = 0,
		TYPE_FLOAT2		 = 1,
		TYPE_FLOAT3		 = 2,
		TYPE_FLOAT4		 = 3,
		TYPE_D3DCOLOR	 = 4,
		TYPE_UBYTE4		 = 5,
		TYPE_SHORT2		 = 6,
		TYPE_SHORT4		 = 7,
		TYPE_UBYTE4N		= 8,
		TYPE_SHORT2N		= 9,
		TYPE_SHORT4N		= 10,
		TYPE_USHORT2N	 = 11,
		TYPE_USHORT4N	 = 12,
		TYPE_UDEC3			= 13,
		TYPE_DEC3N			= 14,
		TYPE_FLOAT16_2	= 15,
		TYPE_FLOAT16_4	= 16,
		TYPE_UNUSED		 = 17
	};

	enum vertex_element_method
	{ 
		METHOD_DEFAULT					 = 0,
		METHOD_PARTIALU					= 1,
		METHOD_PARTIALV					= 2,
		METHOD_CROSSUV					 = 3,
		METHOD_UV								= 4,
		METHOD_LOOKUP						= 5,
		METHOD_LOOKUPPRESAMPLED	= 6
	};

	enum vertex_element_usage
	{ 
		USAGE_POSITION			= 0,
		USAGE_BLENDWEIGHT	 = 1,
		USAGE_BLENDINDICES	= 2,
		USAGE_NORMAL				= 3,
		USAGE_PSIZE				 = 4,
		USAGE_TEXCOORD			= 5,
		USAGE_TANGENT			 = 6,
		USAGE_BINORMAL			= 7,
		USAGE_TESSFACTOR		= 8,
		USAGE_POSITIONT		 = 9,
		USAGE_COLOR				 = 10,
		USAGE_FOG					 = 11,
		USAGE_DEPTH				 = 12,
		USAGE_SAMPLE				= 13
	};

	enum primitive_type
	{
			PT_TRIANGLE_LIST = 0,
			PT_TRIANGLE_STRIP,
			PT_LINE_LIST,
			PT_LINE_STRIP,
			PT_POINT_LIST,
			PT_TRIANGLE_LIST_ADJ,
			PT_TRIANGLE_STRIP_ADJ,
			PT_LINE_LIST_ADJ,
			PT_LINE_STRIP_ADJ,
			PT_QUAD_PATCH_LIST,
			PT_TRIANGLE_PATCH_LIST,
	};

	enum index_type
	{
			INDEX_16 = 0,
			INDEX_32,
	};

	struct header
	{
		std::uint32_t Version;
		std::uint8_t IsBigEndian;
		std::uint64_t HeaderSize;
		std::uint64_t NonBufferDataSize;
		std::uint64_t BufferDataSize;

		std::uint32_t NumVertexBuffers;
		std::uint32_t NumIndexBuffers;
		std::uint32_t NumMeshes;
		std::uint32_t NumTotalSubsets;
		std::uint32_t NumFrames;
		std::uint32_t NumMaterials;

		std::uint64_t VertexStreamHeadersOffset;
		std::uint64_t IndexStreamHeadersOffset;
		std::uint64_t MeshDataOffset;
		std::uint64_t SubsetDataOffset;
		std::uint64_t FrameDataOffset;
		std::uint64_t MaterialDataOffset;
	};

	struct vertex_element
	{
		std::uint16_t Stream;
		std::uint16_t Offset;
		std::uint8_t Type;
		std::uint8_t Method;
		std::uint8_t Usage;
		std::uint8_t UsageIndex;
	};

	struct vertex_buffer
	{
		std::uint64_t NumVertices;
		std::uint64_t SizeBytes;
		std::uint64_t StrideBytes;
		vertex_element Decl[MAX_VERTEX_ELEMENTS];
		std::uint64_t DataOffset;
	};

	struct index_buffer
	{
		std::uint64_t NumIndices;
		std::uint64_t SizeBytes;
		std::uint32_t IndexType;
		std::uint64_t DataOffset;
	};

	struct mesh
	{
		char Name[MAX_MESH_NAME];
		std::uint8_t NumVertexBuffers;
		std::uint32_t VertexBuffers[MAX_VERTEX_STREAMS];
		std::uint32_t IndexBuffer;
		std::uint32_t NumSubsets;
		std::uint32_t NumFrameInfluences;

		float BoundingBoxCenter[3];
		float BoundingBoxExtents[3];

		std::uint64_t SubsetOffset;
		std::uint64_t FrameInfluenceOffset;
	};

	struct subset
	{
		char Name[MAX_SUBSET_NAME];
		std::uint32_t MaterialID;
		std::uint32_t PrimitiveType;
		std::uint64_t IndexStart;
		std::uint64_t IndexCount;
		std::uint64_t VertexStart;
		std::uint64_t VertexCount;
	};

	struct material
	{
		char Name[MAX_MATERIAL_NAME];

		char MaterialInstancePath[MAX_MATERIAL_PATH];

		char DiffuseTexture[MAX_TEXTURE_NAME];
		char NormalTexture[MAX_TEXTURE_NAME];
		char SpecularTexture[MAX_TEXTURE_NAME];

		float Diffuse[4];
		float Ambient[4];
		float Specular[4];
		float Emissive[4];
		float Power;
	};

	template <typename T>
	void readVertexAttribute(const char* buffer, size_t buffer_stride, T* dest, size_t dest_stride, size_t num_vertices)
	{
		for (size_t i = 0; i < num_vertices; ++i)
		{
			*dest = *reinterpret_cast<const T*>(buffer);
			buffer += buffer_stride;
			dest = reinterpret_cast<T*>(reinterpret_cast<char*>(dest) + dest_stride);
		}
	}

	void read(SceneBuilder& builder, const char* begin, size_t length)
	{
		const sdkmesh::header& header = *reinterpret_cast<const sdkmesh::header*>(begin);

		if (header.Version != 101 || header.IsBigEndian != 0)
			throw std::runtime_error("unsupported .sdkmesh");

		const sdkmesh::vertex_buffer* vertex_buffers = reinterpret_cast<const sdkmesh::vertex_buffer*>(begin + header.VertexStreamHeadersOffset);
		const sdkmesh::index_buffer * index_buffers = reinterpret_cast<const sdkmesh::index_buffer*>(begin + header.IndexStreamHeadersOffset);
		const sdkmesh::mesh* meshes = reinterpret_cast<const sdkmesh::mesh*>(begin + header.MeshDataOffset);
		const sdkmesh::subset* subsets = reinterpret_cast<const sdkmesh::subset*>(begin + header.SubsetDataOffset);
		const sdkmesh::material* materials = reinterpret_cast<const sdkmesh::material*>(begin + header.MaterialDataOffset);

		std::vector<vertex> vertices;

		std::unordered_map<const sdkmesh::mesh*, std::uint32_t, std::uint32_t (*)(const sdkmesh::mesh*), bool(*)(const sdkmesh::mesh*, const sdkmesh::mesh*)> vertex_buffer_map {
			10,
			[](const sdkmesh::mesh* mesh)
			{
				return std::accumulate(mesh->VertexBuffers, mesh->VertexBuffers + mesh->NumVertexBuffers, std::uint32_t(0U), std::bit_xor<std::uint32_t>());
			},
			[](const sdkmesh::mesh* a, const sdkmesh::mesh* b)
			{
				return a->NumVertexBuffers != b->NumVertexBuffers ? false : std::equal(a->VertexBuffers, a->VertexBuffers + a->NumVertexBuffers, b->VertexBuffers);
			}
		};

		for (size_t i = 0; i < header.NumMeshes; ++i)
		{
			const sdkmesh::mesh& mesh = meshes[i];

			auto vb = vertex_buffer_map.find(&mesh);

			std::uint32_t buffer_offset;
			if (vb != end(vertex_buffer_map))
			{
				buffer_offset = vb->second;
			}
			else
			{
				buffer_offset = static_cast<std::uint32_t>(vertices.size());
				vertex_buffer_map.insert(std::make_pair(&mesh, buffer_offset));

				auto num_vertices = vertex_buffers[*std::min_element(mesh.VertexBuffers, mesh.VertexBuffers + mesh.NumVertexBuffers, [vertex_buffers](std::uint32_t a, std::uint32_t b) { return vertex_buffers[a].NumVertices < vertex_buffers[b].NumVertices; })].NumVertices;

				vertices.resize(vertices.size() + num_vertices);
				vertex* dest = &vertices[0] + buffer_offset;

				for (size_t j = 0; j < mesh.NumVertexBuffers; ++j)
				{
					const sdkmesh::vertex_buffer& vertex_buffer = vertex_buffers[mesh.VertexBuffers[j]];

					for (int k = 0; k < MAX_VERTEX_ELEMENTS && vertex_buffer.Decl[k].Stream != 255U; ++k)
					{
						switch (vertex_buffer.Decl[k].Usage)
						{
							case USAGE_POSITION:
								if (vertex_buffer.Decl[k].Type == TYPE_FLOAT3 && vertex_buffer.Decl[k].UsageIndex == 0 && vertex_buffer.Decl[k].Method == METHOD_DEFAULT)
								{
									readVertexAttribute<math::float3>(begin + vertex_buffer.DataOffset + vertex_buffer.Decl[k].Offset, vertex_buffer.StrideBytes, &(dest->p), sizeof(vertex), vertex_buffer.NumVertices);
								}
								break;

							case USAGE_TEXCOORD:
								if (vertex_buffer.Decl[k].Type == TYPE_FLOAT2 && vertex_buffer.Decl[k].UsageIndex == 0 && vertex_buffer.Decl[k].Method == METHOD_DEFAULT)
								{
									readVertexAttribute<math::float2>(begin + vertex_buffer.DataOffset + vertex_buffer.Decl[k].Offset, vertex_buffer.StrideBytes, &(dest->t), sizeof(vertex), vertex_buffer.NumVertices);
								}
								break;

							case USAGE_NORMAL:
								if (vertex_buffer.Decl[k].Type == TYPE_FLOAT3 && vertex_buffer.Decl[k].UsageIndex == 0 && vertex_buffer.Decl[k].Method == METHOD_DEFAULT)
								{
									readVertexAttribute<math::float3>(begin + vertex_buffer.DataOffset + vertex_buffer.Decl[k].Offset, vertex_buffer.StrideBytes, &(dest->n), sizeof(vertex), vertex_buffer.NumVertices);
								}
								break;
						}
					}
				}
			}

			const std::uint32_t* subset_indices = reinterpret_cast<const std::uint32_t*>(begin + mesh.SubsetOffset);

			const sdkmesh::index_buffer& index_buffer = index_buffers[mesh.IndexBuffer];

			for (size_t j = 0; j < mesh.NumSubsets; ++j)
			{
				const sdkmesh::subset& subset = subsets[subset_indices[j]];

				if (subset.PrimitiveType == PT_TRIANGLE_LIST)
				{
					std::vector<std::uint32_t> indices(subset.IndexCount);

					if (index_buffer.IndexType == INDEX_16)
					{
						const std::uint16_t* ind = reinterpret_cast<const std::uint16_t*>(begin + index_buffer.DataOffset);
						for (size_t i = 0; i < subset.IndexCount; ++i)
							indices[i] = ind[i] + buffer_offset;
					}
					else if (index_buffer.IndexType == INDEX_32)
					{
						const std::uint32_t* ind = reinterpret_cast<const std::uint32_t*>(begin + index_buffer.DataOffset);
						for (size_t i = 0; i < subset.IndexCount; ++i)
							indices[i] = ind[i] + buffer_offset;
					}
					else
						throw std::runtime_error("invalid index buffer format");

					builder.addSurface(PrimitiveType::TRIANGLES, std::move(indices), subset.Name, std::strlen(subset.Name), nullptr);
					indices.clear();
				}
				else
					throw std::runtime_error("unsupported primitive type");
			}
		}
		
		builder.addVertices(std::move(vertices));
	}
}
