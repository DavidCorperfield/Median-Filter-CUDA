#include <iostream>
#include <string>
#include <tuple>

#include "../include/reader.hpp"

using namespace std;

int main(int argc, char ** argv) {
    Reader reader;
    tuple<uint8_t, char *, char *> input_params = reader.check_command_line(argc, argv);

    uint8_t filter_size = get<0>(input_params);
    char * image_path = get<1>(input_params);
    char * output_path = get<2>(input_params);
    cout << "Image: " << image_path << " has been loaded." << endl;

    // Load image from disk
    // float * header_data = nullptr;
    // unsigned int width, height;
    // sdkLoadPGM(image_path, &header_data, &width, &height);

    // cout << "Width: " << width << "Height: " << height;

    return 0;
}
