// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// portability
#include <portinfo>
// externals
#include <sstream>
// local forward declarations
#include "forward.h"
// my parts
#include "MemoryMap.h"

// meta-methods
pyre::memory::MemoryMap::
MemoryMap(uri_type uri, bool writable, size_type bytes, size_type offset, bool preserve) :
    _uri {uri},
    _info {},
    _bytes(bytes),
    _buffer(0)
{
    // make a channel
    pyre::journal::debug_t channel("pyre.memory.direct");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "constructor: "
        << pyre::journal::newline
        << "    uri: '" << _uri << "'" << pyre::journal::newline
        << "    writable: " << writable << pyre::journal::newline
        << "    bytes: " << bytes << pyre::journal::newline
        << "    offset: " << offset << pyre::journal::newline
        << "    preserve: " << preserve << pyre::journal::newline
        << "    buffer: " << _buffer << pyre::journal::endl;

    // if no filename were given
    if (uri.empty()) {
        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "no uri given"
            << pyre::journal::endl;
        // nothing further to do
        return;
    }

    // compute the desired file size
    size_t desired = offset + bytes;

    // otherwise, ask the filesystem
    int status = ::stat(_uri.data(), &_info);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "looking for '" << _uri << "': status=" << status
        << pyre::journal::endl;

    // if this failed
    if (status) {
        // the only case we handle is the file not existing; complain about everything else
        if (errno != ENOENT) {
            // create a channel
            pyre::journal::error_t error("pyre.memory.direct");
            // complain
            error
                // where
                << pyre::journal::at(__HERE__)
                // what happened
                << "while opening '" << _uri << "'" << pyre::journal::newline
                // why it happened
                << "  reason " << errno << ": " << std::strerror(errno)
                // flush
                << pyre::journal::endl;
            // raise an exception
            throw std::system_error(errno, std::system_category());
        }

        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "the file '" << _uri << "' does not exist"
            << pyre::journal::endl;

        // so, the file doesn't exist; if the caller did not specify a desired map size
        if (bytes == 0) {
            // we have a problem
            std::stringstream problem;
            // describe it
            problem << "while creating '" << uri << "': unknown size";
            // create a channel
            pyre::journal::error_t error("pyre.memory.direct");
            // complain
            error
                // where
                << pyre::journal::at(__HERE__)
                // what happened
                << problem.str()
                // flush
                << pyre::journal::endl;
            // raise an exception
            throw std::runtime_error(problem.str());
        }

        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "creating file '" << _uri << "'"
            << pyre::journal::endl;
        // if we have size information, create the file
        create(uri, desired);
        // get the file information
        ::stat(_uri.data(), &_info);
        // get the actual file size
        size_type actual = _info.st_size;
        // check that it matches our expectations
        if (actual != desired) {
            // if not, we have a problem
            std::stringstream problem;
            // describe it
            problem
                << "while creating '" << uri << "': file size mismatch: asked for "
                << desired << " bytes, got " << _info.st_size << " bytes instead";
            // create a channel
            pyre::journal::error_t error("pyre.memory.direct");
            // complain
            error
                // where
                << pyre::journal::at(__HERE__)
                // what happened
                << problem.str()
                // flush
                << pyre::journal::endl;
            // raise an exception
            throw std::runtime_error(problem.str());
        }

        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "mapping " << bytes << " bytes over file '" << _uri << "' at offset " << offset
            << pyre::journal::endl;
        // map it
        _buffer = map(uri, bytes, offset, writable);
        // all done
        return;
    }

    // tell me
    channel
        << pyre::journal::at(__HERE__)
        << "the file '" << uri << "' exists already"
        << pyre::journal::endl;
    // the file already exists; let's find its size
    size_type actual = _info.st_size;
    // if the actual size is not big enough to hold our data
    if (actual < desired) {
        // if the user doesn't care about the existing file
        if (!preserve) {
            // show me
            channel
                << pyre::journal::at(__HERE__)
                << "size mismatch: actual: " << actual << ", expected: " << desired
                << pyre::journal::newline
                << "re-creating file '" << _uri << "'"
                << pyre::journal::endl;
            // throw the existing file away and rebuild it
            create(uri, desired);
        // otherwise
        } else {
            // we have a problem
            std::stringstream problem;
            // describe it
            problem
                << "while mapping '" << uri
                << "': the file already exists but it is not big enough: "
                << "actual: " << actual << " bytes, requested: " << desired << " bytes";
            // create a channel
            pyre::journal::error_t error("pyre.memory.direct");
            // complain
            error
                // where
                << pyre::journal::at(__HERE__)
                // what happened
                << problem.str()
                // flush
                << pyre::journal::endl;
            // raise an exception
            throw std::runtime_error(problem.str());
        }
    }

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "mapping " << bytes << " bytes over file '" << _uri << "' at offset " << offset
        << pyre::journal::endl;
    // map it
    _buffer = map(uri, bytes, offset, writable);

    // all done
    return;
}

// class methods
// make a file of a specified size
pyre::memory::MemoryMap::size_type
pyre::memory::MemoryMap::
create(uri_type name, size_type bytes) {
    // we take advantage of the POSIX requirement that writing a byte at a file location past its
    // current size automatically fills all the locations before it with nulls

    // create a file stream
    std::ofstream file(name, std::ofstream::binary);
    // move the file pointer to the desired size
    file.seekp(bytes - 1);
    // make a byte
    char null = 0;
    // write a byte
    file.write(&null, sizeof(null));
    // close the stream
    file.close();

    // make a channel
    pyre::journal::debug_t channel("pyre.memory.direct");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "created '" << name << "' (" << bytes << " bytes)"
        << pyre::journal::endl;

    // all done
    return bytes;
}

// memory map the given file
pyre::memory::MemoryMap::pointer
pyre::memory::MemoryMap::
map(uri_type name, size_type bytes, size_type offset, bool writable) {
    // deduce the mode for opening the file
    auto mode = writable ? O_RDWR : O_RDONLY;
    // open the file using low level IO, since we need its file descriptor
    auto fd = ::open(name.c_str(), mode);
    // verify the file was not opened correctly
    if (fd < 0) {
        // we have a problem; make a channel
        pyre::journal::error_t channel("pyre.memory.direct");
        // complain
        channel
            // where
            << pyre::journal::at(__HERE__)
            // what happened
            << "while opening '" << name << "'" << pyre::journal::newline
            // why it happened
            << "  reason " << errno << ": " << std::strerror(errno)
            // flush
            << pyre::journal::endl;
        // raise an exception
        throw std::system_error(errno, std::system_category());
    }

    // deduce the protection flag
    auto prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
    // map it
    pointer buffer = ::mmap(0, bytes, prot, MAP_SHARED, fd, static_cast<offset_t>(offset));
    // check it
    if (buffer == MAP_FAILED) {
        // create a channel
        pyre::journal::error_t channel("pyre.memory.direct");
        // complain
        channel
            // where
            << pyre::journal::at(__HERE__)
            // what happened
            << "failed to map '" << name << "' onto memory (" << bytes << " bytes)"
            << pyre::journal::newline
            // why it happened
            << "  reason " << errno << ": " << std::strerror(errno)
            // flush
            << pyre::journal::endl;
        // raise an exception
        throw std::bad_alloc();
    }

    // make a channel
    pyre::journal::debug_t channel("pyre.memory.direct");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "mapped " << bytes << " bytes from '" << name << "' into "
        << (writable ? "writable" : "read-only")
        << " memory at " << buffer
        << pyre::journal::endl;

    // clean up: close the file
    close(fd);
    // return the payload
    return buffer;
}


// unmap the given buffer
void
pyre::memory::MemoryMap::
unmap(const pointer buffer, size_type bytes) {
    // make a channel
    pyre::journal::debug_t channel("pyre.memory.direct");

    // tell me
    channel
        << pyre::journal::at(__HERE__)
        << "cleaning up existing map at " << buffer
        << pyre::journal::endl;

    // unmap
    int status = ::munmap(const_cast<void *>(buffer), bytes);

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "unmapped " << bytes << " bytes from " << buffer
        << pyre::journal::endl;

    // check whether the memory was unmapped
    if (status) {
        // make a channel
        pyre::journal::error_t error("pyre.memory.direct");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while unmapping " << bytes << " bytes from " << buffer
            << ": error " << errno << ": " << std::strerror(errno)
            << pyre::journal::endl;
    }

    // all done
    return;
}

// end of file
