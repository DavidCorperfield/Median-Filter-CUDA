#include <iostream>
#include <string>
#include <tuple>

#include "../include/reader.hpp"

using namespace std;

int main(int argc, char ** argv) {
    Reader reader;
    tuple<uint8_t, string, string> input_params = reader.check_command_line(argc, argv);

    uint8_t filter_size = get<0>(input_params);
    string image_path = get<1>(input_params);
    string output_path = get<2>(input_params);

    return 0;
}
