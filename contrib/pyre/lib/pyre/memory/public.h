// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_public_h)
#define pyre_memory_public_h

// forward declaration
#include "forward.h"

// the object model
#include "View.h"
#include "ConstView.h"
#include "Heap.h"
#include "MemoryMap.h"
#include "Direct.h"
#include "ConstDirect.h"

// the implementations
// views over existing memory
#define pyre_memory_View_icc
#include "View.icc"
#undef pyre_memory_View_icc

// views over existing const memory
#define pyre_memory_ConstView_icc
#include "ConstView.icc"
#undef pyre_memory_ConstView_icc

// dynamically allocated memory
#define pyre_memory_Heap_icc
#include "Heap.icc"
#undef pyre_memory_Heap_icc

// support for memory backed by files
#define pyre_memory_MemoryMap_icc
#include "MemoryMap.icc"
#undef pyre_memory_MemoryMap_icc

#define pyre_memory_Direct_icc
#include "Direct.icc"
#undef pyre_memory_Direct_icc

#define pyre_memory_ConstDirect_icc
#include "ConstDirect.icc"
#undef pyre_memory_ConstDirect_icc

#endif

// end of file
