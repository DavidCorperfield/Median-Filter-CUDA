#ifndef _reader_hpp
#define _reader_hpp

#include <exception>
#include <iostream>
// #include <fstream>
#include <stdio.h>
#include <stdexcept>
#include <string>
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

class Reader {
public:
    /* Will initialize pgm_source and pgm_destination to nullptr */
    Reader();

    ~Reader();

    /**
     * Command line should take three inputs:
     *     1) An integer filter size (3, 7, 11, or 15)
     *     2) An input filename
     *     3) An output filename
     *
     * Throws a runtime error if none of these things are right.
     */
    std::tuple<uint8_t, char *, char *> check_command_line(int argc, char ** argv);

    /**
     * Loads the PGM input image.
     * Returns the (height, width) of the image.
     */
    std::pair<uint, uint> load_image(const char * image_input_path);

    /**
     * Save the PGM image we did the Median Filter on.
     */
    void save_image(const char * output_path, const uint height, const uint width);

    /* Should probably make these private later on, since good OOP practices and all */
    uchar * pgm_source;
    uchar * pgm_destination;
};


#endif
