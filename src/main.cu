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

/**
    1. The command line should take a) an integer filter size {3, 7, 11, or 15}; b) an input file name; and c) an output file
    2. The input file should be loaded into a data buffer (the file is one byte per pixel).
    3. A timer should mark this time
    4. The file data should be pushed onto the device memory.
    5. The logical structure for processing should be defined
    6. The kernel should be launched
    7. The output data should be copied from device to main memory
    8. The timer should be stopped to capture the device copy-compute-copy time
    9. A golden standard host code version of the answer should be produced and compared to the device generated output
    10. The output should be written to file in PGM format.
    11. The percent of correctness between the host’s golden standard and the device created answer should be reported
    12. The timing statistics should be reported.
 */
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

    const double copy_compute_copy_time = filter.median_filter_gpu(filter_size, reader.pgm_source, reader.pgm_destination, height, width);
    const double error_percentage       = filter.median_filter_verify_errors(filter_size, reader.pgm_source, reader.pgm_destination, height, width);

    /* Save the GPU output image. */
    reader.save_image(image_output_path, height, width);

    /* Total time using GPU AND CPU version (i.e. with the verification). */
    const double total_time = filter.get_timer_value();
    cout << "Copy-compute-copy time is: " << copy_compute_copy_time << " milliseconds " << endl;
    cout << "Total time is: " << total_time << " milliseconds." << endl;
    cout << "Error percentage is: " << error_percentage << endl;

    /* Destroy the timer object. */
    filter.stop_timer();

    return 0;
}
