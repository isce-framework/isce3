# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre.parsing
import pyre.tracking
import pyre.patterns


# the parser
class Parser(pyre.parsing.parser):
    """
    A simple parser for {pfg} files

    This parser understands a file format that uses whitespace to capture the hierarchical
    structure of the input
    """


    # the configuration object model
    from .Assignment import Assignment
    from .Configuration import Configuration
    from .Section import Section


    # interface
    def parse(self, uri, stream, locator):
        """
        Harvest the configuration events in {stream}
        """
        # reset the stack of open containers
        self.context = [ self.Configuration() ]
        # and the list of errors
        self.errors = []

        # set up my processor
        processor = self.processor(locator=locator)
        # tokenize the {stream}
        self.scanner.pyre_tokenize(uri=uri, stream=stream, client=processor)

        # get the top level event container
        configuration = self.context.pop()
        # if all went well, that's all there was
        assert len(self.context) == 0

        # return the harvested events
        yield from configuration.events()
        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # get my scanner
        scanner = self.scanner
        # build the production table
        self.productions = {
            # the ignorables
            scanner.start: self.ignore,
            scanner.comment: self.ignore,
            scanner.whitespace: self.ignore,
            scanner.finish: self.ignore,

            # hierarchy
            scanner.push: self.ignore,
            scanner.pop: self.pop,

            # context specifier
            scanner.section: self.section,
            # assignments
            scanner.assignment: self.assignment,
            scanner.cdata: self.value
            }

        # all done
        return


    # implementation details
    @pyre.patterns.coroutine
    def processor(self, locator):
        """
        Receive tokens from the scanner and handle them
        """
        # for ever
        while True:
            # attempt to
            try:
                # get a token
                token = yield
            # if anything goes wrong
            except self.ParsingError as error:
                # save the error
                self.errors.append(error)
                # and move on
                continue

            # if we retrieved a well formed token, attempt to
            try:
                # look up the relevant production based on this terminal
                production = self.productions[type(token)]
            # if i don't have a production for this token
            except KeyError:
                # it must be a syntax error; build a locator
                loc = pyre.tracking.chain(this=token.locator, next=locator)
                # and an error
                error = self.SyntaxError(token=token, locator=loc)
                # save it
                self.errors.append(error)
                # move on
                continue
            # if all goes well, invoke the production
            production(token=token)
        # all done
        return


    def ignore(self, **kwds):
        """
        Do nothing
        """
        # there is nothing to do here
        return


    def section(self, token):
        """
        Process a section fragment and use it to specify the assignment context
        """
        # make a new section
        section = self.Section(token)
        # push on the stack
        self.context.append(section)
        # all done
        return


    def assignment(self, token):
        """
        Process a key assignment
        """
        # make a new assignment handler
        assignment = self.Assignment(token)
        # push it on the stack
        self.context.append(assignment)
        # collect the value
        self.scanner.pyre_cdata()
        # all done
        return


    def value(self, token):
        """
        Process the value portion of an assignment
        """
        # pop the assignment off the stack
        assignment = self.context.pop()
        # bind the value
        assignment.bind(token)
        # grab the parent context
        parent = self.context[-1]
        # and tell the assignment its processing is finished
        assignment.notify(parent=parent)
        # all done
        return


    def pop(self, token):
        """
        Process the closing of a block
        """
        # get the current section
        section = self.context.pop()
        # and its parent
        parent = self.context[-1]
        # introduce them to each other
        section.notify(parent=parent)
        # all done
        return


    def handleError(self, description, locator):
        """
        Process the {error}
        """
        # build a parsing error with the given {description} and {locator}
        error = self.ParsingError(description=description, locator=locator)
        # save it
        self.errors.append(error)
        # all done
        return


    # private types
    from .Scanner import Scanner as lexer # my superclass uses this to instantiate my scanner

    # private data
    errors = None # the list of parsing and lexing errors
    productions = None # the table of token handlers
    configuration = None # the list of configuration events harvested from the input source


# end of file
