#ifndef _filter_hpp
#define _filter_hpp

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "cuda_helpers/helper_functions.h"
#include "cuda_helpers/helper_cuda.h"
#include <iostream>

/* Grid and Block definitions. Alter these as you please to tweak results. */
#define GRID_X 16
#define GRID_Y 16
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
     * Returns the time taken to do the Median Filter.
     */
    double median_filter_gpu(const uint filter_size, const uchar * input, uchar * output, const uint height, const uint width);

    void median_filter_cpu(const uint filter_size, const uchar * input, uchar * output, const uint height, const uint width);

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
