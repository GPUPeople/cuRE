


#include <CUDA/error.h>
#include <CUDA/array.h>


namespace CU
{
	CUDA_ARRAY_DESCRIPTOR initArray1DDescriptor(std::size_t width, CUarray_format format, unsigned int num_channels)
	{
		return { width, 0U, format, num_channels };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArray1DDescriptor(std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, 0U, 0U, format, num_channels, flags };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArray1DLayeredDescriptor(std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, 0U, layers, format, num_channels, CUDA_ARRAY3D_LAYERED | flags };
	}

	CUDA_ARRAY_DESCRIPTOR initArray2DDescriptor(std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels)
	{
		return { width, height, format, num_channels };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArray2DDescriptor(std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width,  height, 0U, format, num_channels, flags };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArray2DLayeredDescriptor(std::size_t width, std::size_t height, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, height, layers, format, num_channels, CUDA_ARRAY3D_LAYERED | flags };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArrayCubeDescriptor(std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, width, 6U, format, num_channels, CUDA_ARRAY3D_CUBEMAP | flags };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArrayCubeLayeredDescriptor(std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, width, 6U * layers, format, num_channels, CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_LAYERED | flags };
	}

	CUDA_ARRAY3D_DESCRIPTOR initArray3DDescriptor(std::size_t width, std::size_t height, std::size_t depth, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		return { width, height, depth, format, num_channels, flags };
	}


	void initArray1DDescriptor(CUDA_ARRAY_DESCRIPTOR& desc, std::size_t width, CUarray_format format, unsigned int num_channels)
	{
		desc = initArray1DDescriptor(width, format, num_channels);
	}

	void initArray1DDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArray1DDescriptor(width, format, num_channels, flags);
	}

	void initArray1DLayeredDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArray1DLayeredDescriptor(width, layers, format, num_channels, flags);
	}

	void initArray2DDescriptor(CUDA_ARRAY_DESCRIPTOR& desc, std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels)
	{
		desc = initArray2DDescriptor(width, height, format, num_channels);
	}

	void initArray2DDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArray2DDescriptor(width, height, format, num_channels, flags);
	}

	void initArray2DLayeredDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, std::size_t height, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArray2DLayeredDescriptor(width, height, layers, format, num_channels, flags);
	}

	void initArrayCubeDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArrayCubeDescriptor(width, format, num_channels, flags);
	}

	void initArrayCubeLayeredDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArrayCubeLayeredDescriptor(width, layers, format, num_channels, flags);
	}

	void initArray3DDescriptor(CUDA_ARRAY3D_DESCRIPTOR& desc, std::size_t width, std::size_t height, std::size_t depth, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		desc = initArray3DDescriptor(width, height, depth, format, num_channels, flags);
	}


	unique_array createArray(const CUDA_ARRAY_DESCRIPTOR& desc)
	{
		CUarray array;
		succeed(cuArrayCreate(&array, &desc));
		return unique_array(array);
	}

	unique_array createArray(const CUDA_ARRAY3D_DESCRIPTOR& desc)
	{
		CUarray array;
		succeed(cuArray3DCreate(&array, &desc));
		return unique_array(array);
	}

	unique_mipmapped_array createArrayMipmapped(const CUDA_ARRAY3D_DESCRIPTOR& desc, unsigned int levels)
	{
		CUmipmappedArray array;
		succeed(cuMipmappedArrayCreate(&array, &desc, levels));
		return unique_mipmapped_array(array);
	}


	unique_array createArray1D(std::size_t width, CUarray_format format, unsigned int num_channels)
	{
		CUDA_ARRAY_DESCRIPTOR desc;
		initArray1DDescriptor(desc, width, format, num_channels);
		return createArray(desc);
	}

	unique_array createArray1D(std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray1DDescriptor(desc, width, format, num_channels, flags);
		return createArray(desc);
	}

	unique_array createArray1DLayered(std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray1DLayeredDescriptor(desc, width, layers, format, num_channels, flags);
		return createArray(desc);
	}

	unique_mipmapped_array createArray1DMipmapped(std::size_t width, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray1DDescriptor(desc, width, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}

	unique_mipmapped_array createArray1DLayeredMipmapped(std::size_t width, unsigned int layers, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray1DLayeredDescriptor(desc, width, layers, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}


	unique_array createArray2D(std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels)
	{
		CUDA_ARRAY_DESCRIPTOR desc;
		initArray2DDescriptor(desc, width, height, format, num_channels);
		return createArray(desc);
	}

	unique_array createArray2D(std::size_t width, std::size_t height, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray2DDescriptor(desc, width, height, format, num_channels, flags);
		return createArray(desc);
	}

	unique_array createArray2DLayered(std::size_t width, std::size_t height, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray2DLayeredDescriptor(desc, width, height, layers, format, num_channels, flags);
		return createArray(desc);
	}

	unique_mipmapped_array createArray2DMipmapped(std::size_t width, std::size_t height, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray2DDescriptor(desc, width, height, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}

	unique_mipmapped_array createArray2DLayeredMipmapped(std::size_t width, std::size_t height, unsigned int layers, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray2DLayeredDescriptor(desc, width, height, layers, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}


	unique_array createArrayCube(std::size_t width, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArrayCubeDescriptor(desc, width, format, num_channels, flags);
		return createArray(desc);
	}

	unique_array createArrayCubeLayered(std::size_t width, unsigned int layers, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArrayCubeLayeredDescriptor(desc, width, layers, format, num_channels, flags);
		return createArray(desc);
	}

	unique_mipmapped_array createArrayCubeMipmapped(std::size_t width, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArrayCubeDescriptor(desc, width, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}

	unique_mipmapped_array createArrayCubeLayeredMipmapped(std::size_t width, unsigned int layers, unsigned int levels, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArrayCubeLayeredDescriptor(desc, width, layers, format, num_channels, flags);
		return createArrayMipmapped(desc, levels);
	}


	unique_array createArray3D(std::size_t width, std::size_t height, std::size_t depth, CUarray_format format, unsigned int num_channels, unsigned int flags)
	{
		CUDA_ARRAY3D_DESCRIPTOR desc;
		initArray3DDescriptor(desc, width, height, depth, format, num_channels, flags);
		return createArray(desc);
	}
}
