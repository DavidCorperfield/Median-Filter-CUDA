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

/**
 * Gets the global thread index of a 2D Grid of 2D blocks.
 */
__device__
inline uint get_global_thread_index(const uint width) {
    const uint x_index      = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y_index      = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixel_one_d  = x_index + y_index * width;
    return pixel_one_d;
}

__global__
void kernel_median_filter(const uint filter_size, const uchar * device_input_data, uchar * device_output_data, const uint height, const uint width) {
    const uint offset        = (filter_size - 1) / 2;
    const uint filter_length = filter_size * filter_size;
    const uint thread_index  = get_global_thread_index(width);
    const uint x             = thread_index / width;
    const uint y             = thread_index % width;

    // Allocate memory for the filter array
    uchar * filter_array     = new uchar[filter_length];

    // Init the filter array with 0 or 255 values
    // Will write over the indices that are VIEWABLE from the context pixel
#pragma unroll
    for (uint i = 0; i < filter_length; ++i) {
        filter_array[i] = i % 2 == 0 ? MIN_RGB_VALUE : MAX_RGB_VALUE;
    }

    const uchar * context  = device_input_data  + thread_index;
    uchar * output_context = device_output_data + thread_index;

    // Populate the filter_array
    uint filter_array_index = 0;

#pragma unroll
    for (int y_offset = -1 * static_cast<int>(offset); y_offset <= static_cast<int>(offset); ++y_offset) {
#pragma unroll
        for (int x_offset = -1 * static_cast<int>(offset); x_offset <= static_cast<int>(offset); ++x_offset) {
            // Handle special cases for when the offset would place us beyond the bounds of the input.
            const int x_focus = x + x_offset;
            const int y_focus = y + y_offset;

            // Check if one of the neighboring pixels of our context pixel is outside the grid
            if (x_focus < 0 || x_focus >= width || y_focus < 0 || y_focus >= height) {
                continue;
            }
            // Otherwise we're not an edge or corner, so we have all of our neighbors
            filter_array[filter_array_index++] = *(context + static_cast<int>(x_offset) + static_cast<int>(width) * static_cast<int>(y_offset));
        }
    }

    // Sort the filter_array.
    // TODO: If had CUDA 7.0, we'd be using Thrust on the device.
    // But, we don't right now, so just do a Selection Sort.
    uchar swap;
    uint min_index;
#pragma unroll
    for (uint i = 0; i < filter_length - 1; ++i) {
        min_index = i;
#pragma unroll
        for (uint j = i + 1; j < filter_length; ++j) {
            if (filter_array[j] < filter_array[min_index])
                min_index = j;
        }
        swap = filter_array[min_index];
        filter_array[min_index] = filter_array[i];
        filter_array[i] = swap;
    }

    // Grab the median. Note that the since we always had odd window sizes,
    // then filter_size * filter_size is always odd as well - so no need to
    // handle special cases for even or odd number for the median.
    *output_context = filter_array[(filter_length - 1) / 2];
}

double Filter::median_filter_gpu(const uint filter_size, const uchar * host_data, uchar * output, const uint height, const uint width) {
    const int size = height * width * sizeof(uchar);

    /* Allocate device memory for the result. */
    /* Note that output to hold the HOST memory has already been allocated for. */
    void * device_input_data  = nullptr;
    void * device_output_data = nullptr;
    checkCudaErrors(cudaMalloc((void **) & device_input_data, size));
    checkCudaErrors(cudaMalloc((void **) & device_output_data, size));

    /* Copy the input data to the device. */
    checkCudaErrors(cudaMemcpy(
            device_input_data,      // dst
            host_data,              // src
            size,                   // count
            cudaMemcpyHostToDevice
    ));

    /* Launch the kernel! */
    dim3 grid(GRID_X, GRID_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    kernel_median_filter<<< grid, block >>>(filter_size, (uchar *) device_input_data, (uchar *) device_output_data, height, width);

    /* In case the kernel had problems, I'd like to know. */
    checkCudaErrors(cudaGetLastError());

    /* At this point, we just need to copy the device output data back to the host memory. */
    checkCudaErrors(cudaMemcpy(
            output,                 // dst
            device_output_data,     // src
            size,                   // count
            cudaMemcpyDeviceToHost
    ));

    cudaFree(device_input_data);
    cudaFree(device_output_data);

    /* Capture the device copy-compute-copy time. */
    return get_timer_value();
}

void Filter::median_filter_cpu(const uint filter_size, const uchar * input, uchar * output, const uint height, const uint width) {
    // How far in any which direction you can go.
    // (-1, -1) (0, -1) (1, -1)
    // (-1,  0) (0,  0) (1,  0)
    // (-1,  1) (0,  1) (1,  1)
    const uint offset = (filter_size - 1) / 2;
    const uint filter_length = filter_size * filter_size;

    // For testing purposes, print out RGB values of original Lena image.
    #ifdef LENA
        for (uint y = 0; y < height; ++y) {
            for (uint x = 0; x < width; ++x) {
                cout << static_cast<int>(input[x + width * y]) << " ";
            }
            cout << endl;
        }
    #endif

    /**
     * Iterate and perform median filter analysis on every pixel.
     * Make outer loop the ys so that successive reads are as close to each other as possible,
     * i.e. for single-threaded CPU code, it is most important for caching, but for GPUs it is
     * most important for coalesced memory access (and maybe caching).
     * If we iterate over the rows first, we have 0 coalescing then.
     */
    for (uint y = 0; y < height; ++y) {
        for (uint x = 0; x < width; ++x) {

            uchar filter_array[filter_length];
            // Init the filter array with 0 or 255 values
            // Will write over the indices that are VIEWABLE from the context pixel
            for (uint i = 0; i < filter_length; ++i) {
                filter_array[i] = i % 2 == 0 ? MIN_RGB_VALUE : MAX_RGB_VALUE;
            }

            // What pixel am I currently looking at
            const uchar * context  = &input[x + width * y];
            uchar * output_context = &output[x + width * y];

    	    // Populate the filter_array.
            uint filter_array_index = 0;

#pragma unroll
            for (int y_offset = -1 * static_cast<int>(offset); y_offset <= static_cast<int>(offset); ++y_offset) {
#pragma unroll
                for (int x_offset = -1 * static_cast<int>(offset); x_offset <= static_cast<int>(offset); ++x_offset) {
        		    // Handle special cases for when the offset would place us beyond the bounds of the input.
                    const int x_focus = x + x_offset;
                    const int y_focus = y + y_offset;

                    // Check if one of the neighboring pixels of our context pixel is outside the grid
                    if (x_focus < 0 || x_focus >= width || y_focus < 0 || y_focus >= height) {
                        continue;
                    }
                    // Otherwise we're not an edge or corner, so we have all of our neighbors
                    filter_array[filter_array_index++] = *(context + static_cast<int>(x_offset) + static_cast<int>(width) * static_cast<int>(y_offset));
        		}
    	    }

    	    // Sort the filter_array.
            sort(filter_array, filter_array + filter_length);

            // Print the filter array to test.
            // #ifdef _DEBUG
            //     for (uint i = 0; i < filter_length - 1; ++i) {
            //         cout << static_cast<int>(filter_array[i]) << " ";
            //     }
            //     cout << static_cast<int>(filter_array[filter_length - 1]) << endl;
            // #endif

            // Grab the median. Note that the since we always had odd window sizes,
            // then filter_size * filter_size is always odd as well - so no need to
            // handle special cases for even or odd number for the median.
            *output_context = filter_array[(filter_length - 1) / 2];
	   }
    }
}

double Filter::median_filter_verify_errors(const uint filter_size, const uchar * input_data, const uchar * compare, const uint height, const uint width) {
    uchar * cpu_results = (uchar * ) malloc(height * width * sizeof(uchar));
    if (!cpu_results) {
        throw runtime_error("Problems in reserving memory for the CPU version.");
    }

    /* Do the Median Filter using the CPU. */
    median_filter_cpu(filter_size, input_data, cpu_results, height, width);

    const char * cpu_saved_file = "cpu_output.pgm";

    if (!sdkSavePGM<uchar>(cpu_saved_file, cpu_results, width, height)) {
        throw runtime_error("Error in saving the output image!");
    }
    cout << "Using the CPU version, we saved the image with filename: " << cpu_saved_file << endl;

    /* Walk through and compare the pixels of the images to see how many are wrong. */
    uint error_pixel_count = 0;
    for (uint i = 0; i < height * width; ++i) {
        if (cpu_results[i] != compare[i])
            ++error_pixel_count;
    }

    /* Return the percentage of how many pixels are wrong. */
    return error_pixel_count / (height * width);
}
