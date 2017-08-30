//
// Author: Joshua Cohen
// Copyright 2017
//

#include <chrono>
#include <ctime>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include "isce/core/DateTime.h"
using isce::core::DateTime;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::invalid_argument;
using std::istringstream;
using std::get_time;
using std::gmtime;
using std::mktime;
using std::put_time;
using std::regex;
using std::regex_match;
using std::stod;
using std::string;
using std::stringstream;
using std::time_t;
using std::tm;
using std::to_string;

DateTime& DateTime::operator=(const string &dts) {
    /*
     *  Assignment operator for passing in a string. Note that this is usually a challenging
     *  problem to guarantee perfection with, but this will come pretty close to handling the
     *  input properly. Note that the string only needs to match the ISO UTC format, this
     *  assignment cannot guarantee a valid string beyond the pattern needed to parse properly.
     */

    // Generate regex pattern to match ISO-formatted string of YYYY-MM-DDThh:mm:SS.s (see 
    // ::toIsoString()'s docstring for further info)
    regex iso_pattern("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]*");
    // Check input string against regex. Assume we're okay to parse if the formatting is ISO-
    // compatible
    if (!regex_match(dts, iso_pattern)) {
        string errstr = "Error: String passed to DateTime constructor does not match ISO-";
        errstr += "compatible UTC formatting.";
        throw invalid_argument(errstr);
    }

    // If we make it here assume that we can at least naively parse the input. Do the inverse
    // of the operations we do to convert from chrono::time_point to string. Here we know there
    // are 19 characters in the ISO-formatted string before we reach the fractional second, so
    // we only pull that part of the string
    double fractional = stod(dts.substr(19));
    // C-type time structs only accept ISO-formatted times without the fractional part, so create a
    // quick copy of the substring without the fractional part (i.e. the first 19 characters)
    string c_formatted(dts, 0, 19);
    // Create an empty C-type tm struct
    tm c_datetime;
    // Initialize an istringstream with the contents of the C-formatted string
    istringstream isstrm(c_formatted);
    // Parse the input string using the get_time function and the same formatting string we use in
    // ::toIsoString()
    isstrm >> get_time(&c_datetime, "%Y-%m-%dT%H:%M:%S");
    // Initialize internal chrono::time_point to the time contained in the input string using
    // chrono's from_time_t()
    t = system_clock::from_time_t(mktime(&c_datetime));
    // Add the fractional part back using the same methods that the overloaded operators do
    t += duration_cast<system_clock::duration>(duration<double>(fractional));
    return *this;
}

DateTime& DateTime::operator=(const double dtd) {
    /*
     *  Simple assignment operator that takes a double, assumes it's expressed in seconds-since-
     *  epoch, and initialize the corresponding internal chrono::time_point to match the input.
     */

    // Re-initialize object to default POSIX epoch point
    t = time_point<high_resolution_clock>();
    // Add the input time as a chrono::duration
    t += duration_cast<system_clock::duration>(duration<double>(dtd));
    return *this;
}

string DateTime::toIsoString() const {
    /*
     *  String formatting is always a tricky subject, so full description of the parsing and
     *  processing is:
     *
     *  ISO date/time string formatting varies between 6 levels of granularity, but we'll assume
     *  for the DateTime object that we will always want to print the finest level (which includes
     *  fractional seconds). The ISO character formatting, given that the std::chrono::time_point
     *  is highly precise, will be:
     *
     *  YYYY-MM-DDThh:mm:SS.s
     *
     *  Where:
     *      YYYY == Year
     *      MM   == Month
     *      DD   == Day
     *      hh   == Hour (24h notation)
     *      mm   == Minute
     *      SS   == Integral second
     *      s    == Fractional second of the format [0-9]{1,16}
     */

    // C-type time_t-parsed copy of internal std::chrono::time_point
    time_t tc = system_clock::to_time_t(t);
    // Create a stringstream object that we can manipulate using put_time
    stringstream sstrm;
    // Store the first part of the ISO-formatted string in the stringstream using put_time. Also
    // uses gmtime to convert the time_t copy of the time_point to a UTC-aligned date+time format.
    // See http://www.cplusplus.com/reference/iomanip/put_time/ for formatting explanation.
    sstrm << put_time(gmtime(&tc), "%Y-%m-%dT%H:%M:%S.");
    // Initialize the return string with the first part of the ISO-formatted string from the stream
    string datetime_str = sstrm.str();
    // Add the fractional second to the string by calculating how many nanoseconds are in the
    // fractional part. We can use time_since_epoch() on the internal time_point since the
    // fractional second is independent of the reference point and the reference is fixed to POSIX
    // epoch besides
    long int nanosec = duration_cast<nanoseconds>(t.time_since_epoch()).count();
    datetime_str += to_string(nanosec % static_cast<long int>(1e9));
    return datetime_str;
}

