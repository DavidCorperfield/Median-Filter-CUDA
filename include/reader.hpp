#ifndef _reader_
#define _reader_

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

class Reader {
public:
    Reader() = default;

    /**
     * Command line should take three inputs:
     *     1) An integer filter size (3, 7, 11, or 15)
     *     2) An input filename
     *     3) An output filename
     *
     * Returns (filter_size, input, output)
     */
    std::tuple<uint8_t, std::string, std::string> check_command_line(int argc, char ** argv);
};


#endif
