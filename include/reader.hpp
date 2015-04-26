#ifndef _reader_
#define _reader_

#include <exception>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Boost Libs
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include "../include/cuda_helpers/helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "../include/cuda_helpers/helper_cuda.h"        // helper functions for CUDA error check

typedef unsigned int uint;
typedef unsigned char uchar;

#define EXPECTED_HEIGHT 512
#define EXPECTED_WIDTH 512

// static uint EXPECTED_WIDTH 512;
// static uint EXPECTED_HEIGHT 512;

class Reader {
public:
    Reader() = default;

    /**
     * Command line should take three inputs:
     *     1) An integer filter size (3, 7, 11, or 15)
     *     2) An input filename
     *     3) An output filename
     *
     * Returns (filter_size, input, output)
     */
    std::tuple<uint8_t, char *, char *> check_command_line(int argc, char ** argv);

    void load_image(const char * image_path);

    uchar * pgm_source;
    uchar * pgm_destination;
};


#endif
