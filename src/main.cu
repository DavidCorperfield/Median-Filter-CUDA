#include <iostream>
#include <string>
#include <tuple>

#include "../include/reader.hpp"

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

int main(int argc, char ** argv) {
    Reader reader;
    tuple<uint8_t, char *, char *> input_params = reader.check_command_line(argc, argv);

    uint8_t filter_size = get<0>(input_params);
    char * image_path = get<1>(input_params);
    char * output_path = get<2>(input_params);

    reader.load_image(image_path);

    return 0;
}
