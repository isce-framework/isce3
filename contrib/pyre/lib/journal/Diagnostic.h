// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Diagnostic_h)
#define pyre_journal_Diagnostic_h

// place Diagnostic in namespace pyre::journal
namespace pyre {
    namespace journal {
        template <typename> class Diagnostic;

        // the injection operator
        template <typename Channel, typename item_t>
        inline
        Diagnostic<Channel> &
        operator<< (Diagnostic<Channel> &, item_t);
    }
}


// declaration
template <typename Severity>
class pyre::journal::Diagnostic : public pyre::journal::Chronicler {
    // types
public:
    using severity_t = Severity;
    using string_t = std::string;

    using entry_t = std::vector<string_t>;
    using buffer_t = std::stringstream;
    using metadata_t = std::map<string_t, string_t>;

    // interface
public:
    // complete an entry
    inline Diagnostic & record();
    // add a new line
    inline Diagnostic & newline();
    // decorate with (key,value) meta data
    inline Diagnostic & setattr(string_t, string_t);
    // inject an item into the message stream
    template <typename item_t>
    inline Diagnostic & inject(item_t datum);

    // meta methods
protected:
    inline ~Diagnostic();
    inline Diagnostic(string_t, string_t);

    // disallow
private:
    Diagnostic(const Diagnostic &) = delete;
    Diagnostic & operator=(const Diagnostic &) = delete;

    // data members
private:
    entry_t _entry;
    buffer_t _buffer;
    metadata_t _metadata;

    // implementation details
protected:
    inline void _startRecording();
    inline void _endRecording();
};


// get the inline definitions
#define pyre_journal_Diagnostic_icc
#include "Diagnostic.icc"
#undef pyre_journal_Diagnostic_icc


# endif
// end of file
