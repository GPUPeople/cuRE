


#ifndef INCLUDED_INSTRUMENTATION
#define INCLUDED_INSTRUMENTATION

#pragma once


namespace Instrumentation
{
	class Clock32
	{
	public:
		typedef unsigned int clock_t;

		__device__
		static clock_t clock()
		{
			return ::clock();
		}
	};

	class Clock64
	{
	public:
		typedef unsigned long long clock_t;

		__device__
		static clock_t clock()
		{
			return ::clock64();
		}
	};


	template <typename T>
	class TimingResult;

	template <>
	class TimingResult<unsigned int>
	{
	public:
		using data_t = ulonglong2;

	private:
		data_t data;

	public:
		__device__
		void reset()
		{
			data.x = data.y = 0ULL;
			//data.x = data.y = data.w = __int_as_float(0);
			//data.z = __int_as_float(0xFFFFFFFFU);
			//data.z = __int_as_float(0);
		}

		__device__
		void update(unsigned int t)
		{
			////data.z += 1.0f;

			////float d = t - data.x;
			////data.x += d / data.z;
			////data.y += d * (t - data.x);
			////data.x = (data.x + t) * 0.5f;
			//unsigned long long time = ull_from_hilo(__float_as_int(data.y), __float_as_int(data.x));
			//time += t;
			//data.x = __int_as_float(lo_from_ull(time));
			//data.y = __int_as_float(hi_from_ull(time));
			////data.z = __int_as_float(min(t, __float_as_int(data.z)));
			////data.w = __int_as_float(max(t, __float_as_int(data.w)));
			//data.w = __int_as_float(__float_as_int(data.w) + 1);
			data.x += t;
			++data.y;
		}

		__device__
		data_t read()
		{
			return data;
		}
	};


	template <int MAX_NUM_BLOCKS, unsigned int NUM_TIMERS, template <unsigned int> class Enable>
	class Instrumentation
	{
		TimingResult<unsigned int> per_block_timings[NUM_TIMERS][MAX_NUM_BLOCKS];

		template <bool ENABLE>
		struct PerBlockTimer
		{
			template <int LEVEL>
			__device__
			static unsigned int& counter()
			{
				__shared__ static unsigned int state;
				return state;
			}

		public:
			template <int LEVEL>
			__device__
			static void begin()
			{
				if (threadIdx.x == 0)
					counter<LEVEL>() = Clock32::clock();
			}

			template <int LEVEL>
			__device__
			static void end(TimingResult<unsigned int>* timings)
			{
				if (threadIdx.x == 0)
					timings[blockIdx.x].update(Clock32::clock() - counter<LEVEL>());
			}
		};

		template <>
		struct PerBlockTimer<false>
		{
			template <int LEVEL>
			__device__ static void begin() {}
			template <int LEVEL>
			__device__ static void end(TimingResult<unsigned int>* timings) {}
		};

	public:
		__device__
		void reset()
		{
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < MAX_NUM_BLOCKS * NUM_TIMERS; i += gridDim.x * blockDim.x)
			{
				int timer = i / MAX_NUM_BLOCKS;
				int timing = i % MAX_NUM_BLOCKS;
				per_block_timings[timer][timing].reset();
			}
		}

		__device__
		void read(TimingResult<unsigned int>::data_t* per_block_timing_buffer, unsigned int buffer_stride, unsigned int num_blocks)
		{
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < MAX_NUM_BLOCKS * NUM_TIMERS; i += gridDim.x * blockDim.x)
			{
				int timer = i / MAX_NUM_BLOCKS;
				int timing = i % MAX_NUM_BLOCKS;

				if (timing < num_blocks)
					per_block_timing_buffer[timer * buffer_stride / 16U + timing] = per_block_timings[timer][timing].read();
			}

			reset();
		}

		template <int id, int LEVEL>
		__device__
		void enter()
		{
			PerBlockTimer<Enable<id>::value>::template begin<LEVEL>();
		}

		template <int id, int LEVEL>
		__device__
		void leave()
		{
			static_assert(id < NUM_TIMERS, "invalid timer id");
			PerBlockTimer<Enable<id>::value>::template end<LEVEL>(per_block_timings[id]);
		}
	};
}

#endif  // INCLUDED_INSTRUMENTATION
