#include "../include/filter.hpp"

using namespace std;

/* ============================================DEVICE INFO============================================
  CUDA Driver Version / Runtime Version          6.5 / 6.5
  CUDA Capability Major/Minor version number:    3.0
  Total amount of global memory:                 4096 MBytes (4294770688 bytes)
  ( 8) Multiprocessors, (192) CUDA Cores/MP:     1536 CUDA Cores
  GPU Clock rate:                                797 MHz (0.80 GHz)
  Memory Clock rate:                             2500 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Bus ID / PCI location ID:           0 / 3
  Compute Mode: < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

  deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 6.5, CUDA Runtime Version = 6.5, NumDevs = 1, Device0 = GRID K520
*/

/* =============== TRY EXPLICIT TEMPLATE INSTANTIATION ====================== */
template double Filter::median_filter_gpu<3>(const uchar *, uchar *, const uint, const uint);
template double Filter::median_filter_gpu<7>(const uchar *, uchar *, const uint, const uint);
template double Filter::median_filter_gpu<11>(const uchar *, uchar *, const uint, const uint);
template double Filter::median_filter_gpu<15>(const uchar *, uchar *, const uint, const uint);

template<uint8_t filter_size>
double Filter::median_filter_gpu(const uchar * host_data, uchar * output, const uint height, const uint width) {
    const uint size = height * width * sizeof(uchar);

    /* Allocate device memory for the result. */
    uchar * device_data = nullptr;
    checkCudaErrors(cudaMalloc((void **) & device_data, size));
    checkCudaErrors(
        cudaMemcpy(
            device_data,    // dst
            host_data,      // src
            size,           // count
            cudaMemcpyHostToDevice
        )
    );

    return 0;
}

template<uint8_t filter_size = 3>
void Filter::median_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width) {
    // How far in any which direction you can go.
    // (-1, -1) (0, -1) (1, -1)
    // (-1,  0) (0,  0) (1,  0)
    // (-1,  1) (0,  1) (1,  1)
    const uint8_t offset = (filter_size - 1) / 2;
    uchar filter_array[filter_size * filter_size];

    // Iterate and perform median filter analysis on every pixel.
    for (uint x = 0; x < width; x++) {
	for (uint y = 0; y < height; y++) {
	    uchar * context        = &input[x + width * y];
	    uchar * output_context = &output[x + width * y];

	    // Populate the filter_array.
	    uint filter_array_index = 0;
	    for (uint x_offset = -offset; x_offset < offset; x_offset++) {
		for (uint y_offset = -offset; y_offset < offset; y++) {
		    // Handle special case for when the offset would place us beyond the bounds of the input.
		    // (a la the edges or corners of an image)
		    //
		    // Well, duh, we have x and y right above.
		    
		    filter_array[filter_array_index++] = *(context + x_offset + width * y_offset);
		}
	    }

	    // Sort the filter_array.
	    // blah.

	    // Grab the median.
	    *output_context = filter_array[(filter_size * filter_size - 1) / 2];
	}
    }
}

inline void Filter::start_timer() {
    StopWatchInterface * timer = nullptr;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
}

inline double Filter::stop_timer(StopWatchInterface * timer) {
    sdkStopTimer(&timer);
    const double time_taken = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    return time_taken;
}
