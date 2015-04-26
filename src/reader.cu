#include "../include/reader.hpp"

using namespace std;

tuple<uint8_t, char *, char *> Reader::check_command_line(int argc, char ** argv) {
    if (argc < 4) {
        cout << "Invalid Usage." << endl;
        cout << "Proper usage is: " << "./mf <filter size> <input_filename> <output_filename> \n" << endl;
        throw runtime_error("Invalid use!");
    }

    uint8_t filter_size = boost::lexical_cast<unsigned int>(argv[1]);
    vector<uint8_t> valid_filters = {3, 7, 11, 15};
    if (find(valid_filters.begin(), valid_filters.end(), filter_size) == valid_filters.end()) {
        throw runtime_error("Invalid filter size! Must be 3, 7, 11, or 15");
    }

    // string input_file_path(argv[2]);
    if (!boost::filesystem::exists(argv[2].to_string())) {
        throw runtime_error("No image to do a median filter on!");
    };

    // string output_file_path(argv[3]);
    if (boost::filesystem::exists(argv[3].to_string())) {
        throw runtime_error("Output file already exists!");
    }

    return make_tuple(filter_size, argv[2], argv[3]);
}
