


#ifndef INCLUDED_INDEXED_TRIANGLE_MESH
#define INCLUDED_INDEXED_TRIANGLE_MESH

#pragma once

#include "Scene.h"


class IndexedTrianglemesh : public Scene
{
public:
	IndexedTrianglemesh();

	IndexedTrianglemesh(const IndexedTrianglemesh&) = delete;
	IndexedTrianglemesh& operator =(const IndexedTrianglemesh&) = delete;

	void save(Config& config) const override {}
};


#endif  // INCLUDED_INDEXED_TRIANGLE_MESH
