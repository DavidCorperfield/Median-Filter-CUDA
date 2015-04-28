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

    /* Check for right number of arguments, correct filter size. Throw runtime error if none of this is correct. */
    tuple<uint8_t, char *, char *> input_parameters = reader.check_command_line(argc, argv);
    uint8_t filter_size = get<0>(input_parameters);
    const char * image_input_path = get<1>(input_parameters);
    char * image_output_path = get<2>(input_parameters);

    pair<uint, uint> image_dimensions = reader.load_image(image_input_path);
    const uint height = get<0>(image_dimensions);
    const uint width = get<1>(image_dimensions);

    /* Do some Median Filter magic. */
    Filter filter;
    double time_taken = 0;

    /**
     * Credit to Will for the idea of the switch statement. This is used since filter size is not known
     * until run time but we need our templates at compile time. So, just hard code the sizes.
     * TODO: Think of better way to extend this in case we had more filter sizes by using
     * explicit template instantiation.
     */
    switch(filter_size) {
        /* Should definitely match one of these cases since I checked for bad filter sizes earlier on. */
        case 3:
            time_taken = filter.median_filter_gpu<3>(reader.pgm_source, reader.pgm_destination, height, width);
            break;
        case 7:
            time_taken = filter.median_filter_gpu<7>(reader.pgm_source, reader.pgm_destination, height, width);
            break;
        case 11:
            time_taken = filter.median_filter_gpu<11>(reader.pgm_source, reader.pgm_destination, height, width);
            break;
        case 15:
            time_taken = filter.median_filter_gpu<15>(reader.pgm_source, reader.pgm_destination, height, width);
            break;
    }
    reader.save_image(image_output_path, height, width);

    return 0;
}
