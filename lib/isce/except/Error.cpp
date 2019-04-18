#include "Error.h"

using namespace isce::except;

// generic error message
std::string errmsg(const SrcInfo& info) {
    return "Error in file " + std::string(info.file) +
           ", line " + std::to_string(info.line) +
           ", function " + info.func;
}

// message with generic prefix
std::string errmsg(const SrcInfo& info, std::string msg) {
    return errmsg(info) + ": " + msg;
}

template<typename T>
Error<T>::Error(const SrcInfo& info) :
    info(info),
    T(errmsg(info)) {}

template<typename T>
Error<T>::Error(const SrcInfo& info, std::string msg) :
    info(info),
    T(errmsg(info, msg)) {}

template class Error<std::length_error>;
template class Error<std::runtime_error>;
