


#ifndef INCLUDED_CUDARASTER_GEOMETRY
#define INCLUDED_CUDARASTER_GEOMETRY

#include <cstdint>
#include <CUDA/memory.h>
#include <Resource.h>
#include "App.hpp"


namespace CUDARaster
{
	class Renderer;


	class IndexedGeometry : public ::Geometry
	{
	protected:
		IndexedGeometry(const IndexedGeometry&) = delete;
		IndexedGeometry& operator =(const IndexedGeometry&) = delete;

		FW::App& app_;

		Buffer in_vertices;
		Buffer out_vertices;
		Buffer in_indices;

		uint32_t num_verts;
		uint32_t num_triangles;

		Renderer* renderer;

	public:
		FW::App* getApp() {return &app_;	}

		IndexedGeometry(Renderer* renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, FW::App& app);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};


	class ClipspaceGeometry : public ::Geometry
	{
	protected:
		ClipspaceGeometry(const ClipspaceGeometry&) = delete;
		ClipspaceGeometry& operator =(const ClipspaceGeometry&) = delete;

		FW::App& app_;

		Buffer in_vertices;
		Buffer out_vertices;
		Buffer in_indices;

		uint32_t num_verts;

		Renderer* renderer;

	public:
		FW::App* getApp() { return &app_; }

		ClipspaceGeometry(Renderer* renderer, const float* position, size_t num_vertices, FW::App& app);

		virtual void draw() const override;
		virtual void draw(int from, int num_indices) const override;
	};

	class EyeCandyGeometry : public ::Geometry
	{
	protected:
		EyeCandyGeometry(const EyeCandyGeometry&) = delete;
		EyeCandyGeometry& operator =(const EyeCandyGeometry&) = delete;

		FW::App& app_;

		Buffer in_vertices;
		Buffer out_vertices;
		Buffer in_indices;

		uint32_t num_indices;
		uint32_t num_verts;

		Renderer* renderer;

	public:
		FW::App* getApp() { return &app_; }

		EyeCandyGeometry(Renderer* renderer, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles, FW::App& app);

		virtual void draw() const override;
		virtual void draw(int from, int num_indices) const override;
	};
}

#endif  // INCLUDED_CUDARASTER_GEOMETRY
