#ifndef _filter_hpp
#define _filter_hpp

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_helpers/helper_functions.h"
#include "cuda_helpers/helper_cuda.h"
#include <iostream>

/* Grid and Block definitions. Alter these as you please to tweak results. */
#ifndef GRID_SET
    #define GRID_X 16
    #define GRID_Y 16
    #define BLOCK_X 32
    #define BLOCK_Y 32
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

class Filter {
public:
    Filter() = default;
    ~Filter() = default;

    /**
     * Does a Median Filter on the input PGM image.
     * Returns the time taken to do the Median Filter.
     */
    template<uint8_t filter_size>
    double median_filter_gpu(uchar * data, uchar * output, uint height, uint width) {
        return filter_size;
    }


    // void test(const uint8_t filter_size, Filter & f, uchar * data, uchar * output, uint height, uint width) {
    //     double time_taken = 0;
    //     time_taken = f.median_filter_gpu<filter_size>(data, output, height, width);
    //     std::cout << "Time taken is: " << time_taken << std::endl;
    // }
};

#endif
