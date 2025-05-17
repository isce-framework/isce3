#pragma once
#include <cstddef>
#include <iterator>
#include <pybind11/pybind11.h>

template <typename T>
class TypedPythonIterator {
public:
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    TypedPythonIterator(pybind11::sequence seq, ssize_t offset = 0)
        : sequence{seq}, index{offset} {}

    reference operator*() {
        // This class exists to perform the Python->C++ cast right here.
        return sequence[index].cast<reference>();
    }

    TypedPythonIterator& operator++() {
        ++index;
        return *this;
    }

    bool operator==(const TypedPythonIterator& other) const {
        return sequence.is(other.sequence) and (index == other.index);
    }

    bool operator!=(const TypedPythonIterator& other) const {
        return !(*this == other);
    }

private:
    pybind11::sequence sequence;
    ssize_t index;
};

template <typename T>
class TypedPythonSequence {
public:
    using value_type = T;
    TypedPythonSequence(const pybind11::sequence& seq) : sequence{seq} {}
    TypedPythonIterator<T> begin() const {
        return TypedPythonIterator<T>(sequence, 0);
    }
    TypedPythonIterator<T> end() const {
        return TypedPythonIterator<T>(sequence, sequence.size());
    }
    auto size() const {
        return sequence.size();
    }

private:
    pybind11::sequence sequence;
};
