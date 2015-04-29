#ifndef _filter_hpp
#define _filter_hpp

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_helpers/helper_functions.h"
#include "cuda_helpers/helper_cuda.h"
#include <iostream>
#include <thrust/sort.h>

/* Grid and Block definitions. Alter these as you please to tweak results. */
#define GRID_X 512
#define GRID_Y 512
#define BLOCK_X 32
#define BLOCK_Y 32

#define MIN_RGB_VALUE 0
#define MAX_RGB_VALUE 255

typedef unsigned char uchar;
typedef unsigned int uint;

class Filter {
public:
    /**
     * Does a Median Filter on the input PGM image.
     * Return the copy-compute-copy time.
     */
    double median_filter_gpu(const uint filter_size, const uchar * input, uchar * output, const uint height, const uint width);

    void median_filter_cpu(const uint filter_size, const uchar * input, uchar * output, const uint height, const uint width);

    /* Determines the percentage of how many pixels are wrong between the CPU and GPU version. */
    double median_filter_verify_errors(const uint filter_size, const uchar * data, const uchar * compare, const uint width, const uint height);

    inline void start_timer() {
        this->timer = nullptr;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
    }

    inline double get_timer_value() {
        return sdkGetTimerValue(&timer);
    }

    /* Stops and deletes the timer object. */
    inline void stop_timer() {
        sdkStopTimer(&timer);
        sdkDeleteTimer(&timer);
    }

    StopWatchInterface * timer;
};

#endif
