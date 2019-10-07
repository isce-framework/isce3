// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// external packages
#include <map>
#include <vector>
#include <string>
#include <sstream>

// my class header
#include "Renderer.h"


// interface
// rendering of diagnostic entries
pyre::journal::Renderer::string_t
pyre::journal::Renderer::
render(
       pyre::journal::Renderer::entry_t & entry,
       pyre::journal::Renderer::metadata_t & metadata)
{
    // build the string stream
    stream_t stream;
    // place the header, the body and the footer
    header(stream, entry, metadata);
    body(stream, entry, metadata);
    footer(stream, entry, metadata);
    // and return the string
    return stream.str();
}


// meta methods
// destructor
pyre::journal::Renderer::
~Renderer() {}


// implementation
// the header
void
pyre::journal::Renderer::
header(
       pyre::journal::Renderer::stream_t & stream,
       pyre::journal::Renderer::entry_t & entry,
       pyre::journal::Renderer::metadata_t & metadata)
{
    string_t marker(" >> ");

    // render the diagnostic severity and channel
    stream
        // a marker
        << marker
        // the severity
        <<  metadata["severity"]
        // the channel name
        << "(" << metadata["channel"] << "): ";

    // get the file name from the metadata
    string_t & filename = metadata["filename"];
    // if we don't know the filename
    if (filename.empty()) {
        // add a newline
        stream << std::endl;
        // and bail
        return;
    }

    // for rendering the filename
    const size_t maxlen = 40;

    // names longer than {maxlen} characters get shortened
    if (filename.size() > maxlen) {
        stream
            << filename.substr(0, maxlen/4 - 3)
            << "..."
            << filename.substr(filename.size() - 3*maxlen/4);
    } else {
        stream << filename;
    }
    // render the line number
    stream << ":" << metadata["line"];
    // and the function name, if there
    string_t & function = metadata["function"];
    if (! function.empty()) {
        stream << ":" << function;
    }
    // add a separator
    stream << ": ";

    // and a newline
    stream << std::endl;

    // and return
    return;
}


// the body
void
pyre::journal::Renderer::
body(
       pyre::journal::Renderer::stream_t & stream,
       pyre::journal::Renderer::entry_t & entry,
       pyre::journal::Renderer::metadata_t & metadata)
{
    // iterate over the strings in {entry}
    for (entry_t::const_iterator line = entry.begin(); line != entry.end(); ++line) {
        // render each one
        stream << " -- " << *line << std::endl;
    }
    // and return
    return;
}


// the footer
void
pyre::journal::Renderer::
footer(
       pyre::journal::Renderer::stream_t & stream,
       pyre::journal::Renderer::entry_t & entry,
       pyre::journal::Renderer::metadata_t & metadata)
{}

// end of file
