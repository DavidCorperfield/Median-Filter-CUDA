#include <exception>
#include <iostream>
#include <stdexcept>
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
    tuple<uint, char *, char *> input_parameters = reader.check_command_line(argc, argv);
    const uint filter_size                       = get<0>(input_parameters);
    const char * image_input_path                = get<1>(input_parameters);
    char * image_output_path                     = get<2>(input_parameters);

    pair<uint, uint> image_dimensions            = reader.load_image(image_input_path);
    const uint height                            = get<0>(image_dimensions);
    const uint width                             = get<1>(image_dimensions);

    /* Do some Median Filter magic. */
    Filter filter;

    filter.start_timer();

    filter.median_filter_gpu(filter_size, reader.pgm_source, reader.pgm_destination, height, width);
    const double error_percentage = filter.median_filter_verify_errors(filter_size, reader.pgm_source, reader.pgm_destination, height, width);

    reader.save_image(image_output_path, height, width);

    const double total_time = filter.get_timer_value();
    cout << "Total time is: " << total_time << " milliseconds." << endl;
    cout << "Error percentage is: " << error_percentage << endl;

    filter.stop_timer();

    return 0;
}
