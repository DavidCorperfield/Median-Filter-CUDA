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

    if (!boost::filesystem::exists(argv[2])) {
        throw runtime_error("No image to do a median filter on!");
    };

    if (boost::filesystem::exists(argv[3])) {
        throw runtime_error("Output file already exists!");
    }

    return make_tuple(filter_size, argv[2], argv[3]);
}

void Reader::load_image(const char * image_path) {
    uint height, width;

    #ifdef _DEBUG
        cout << image_path << endl;
    #endif

    pgm_source = pgm_destination = nullptr;

    /* If we have problems loading the image, or if the height or width is not what we expect (512 px), quit early. */
    if (!sdkLoadPGM<uchar>(image_path, &pgm_source, &width, &height)) {
        throw runtime_error("Problem loading the PGM image!");
    }
    // if (width != EXPECTED_WIDTH || height != EXPECTED_HEIGHT) {
    //     throw runtime_error("Unexpected width or height!");
    // }

    /* Since loadPGM naturally doesn't check the malloc, we should do that. */
    // pgm_destination = (uchar *) malloc(height * width);
    // if (!pgm_destination) {
    //     throw runtime_error("Problem with malloc for the destination image!");
    // }
}
