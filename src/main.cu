#include <iostream>
#include <string>
#include <tuple>

#include "../include/reader.hpp"
#include "../include/filter.hpp"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

int main(int argc, char ** argv) {
    Reader reader;
    tuple<uint8_t, char *, char *> input_params = reader.check_command_line(argc, argv);

    uint8_t filter_size = get<0>(input_params);
    char * image_input_path = get<1>(input_params);
    char * image_output_path = get<2>(input_params);

    pair<uint, uint> image_dimensions = reader.load_image(image_input_path);
    uint height = get<0>(image_dimensions);
    uint width = get<1>(image_dimensions);

    /* Do some Median Filter magic. */
    Filter filter;
    double time_taken = filter.median_filter_gpu<filter_size>(reader.pgm_source, reader.pgm_destination, height, width);

    reader.save_image(image_output_path, height, width);

    return 0;
}
