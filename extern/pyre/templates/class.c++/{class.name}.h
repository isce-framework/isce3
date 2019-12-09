// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
//

// code guard
#if !defined({project.name}_{class.category}_{class.name}_h)
#define {project.name}_{class.category}_{class.name}_h

// place {{{class.name}}} in the proper namespace
namespace {project.name} {{
    namespace {class.category} {{
        class {class.name};
    }}
}}

// declaration
class {project.name}::{class.category}::{class.name} {{
    // types
public:

    // interface
public:

    // meta-methods
public:
    virtual ~{class.name}();
    inline {class.name}();

    // data
private:

    // disallow
private:
    inline {class.name}(const {class.name} &);
    inline const {class.name} & operator=(const {class.name} &);
    }};

// get the inline definitions
#define {project.name}_{class.category}_{class.name}_icc
#include "{class.name}.icc"
#undef {project.name}_{class.category}_{class.name}_icc

# endif
// end of file
