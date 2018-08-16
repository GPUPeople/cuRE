


#ifndef INCLUDED_INDEXED_TRIANGLE_MESH
#define INCLUDED_INDEXED_TRIANGLE_MESH

#pragma once

#include "Scene.h"



class IndexedTrianglemesh : public Scene
{
private:


public:
	IndexedTrianglemesh(const IndexedTrianglemesh&) = delete;
	IndexedTrianglemesh& operator =(const IndexedTrianglemesh&) = delete;

	IndexedTrianglemesh();

	void save(Config& config) const override {}
};


#endif  // INCLUDED_INDEXED_TRIANGLE_MESH
