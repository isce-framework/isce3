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
    A simple parser for {cfg} files

    This parser understands a variant of the windows {INI} file format. See the package
    documentation for details.
    """


    # types
    from .exceptions import ParsingError, SyntaxError
    from ..events import Assignment, ConditionalAssignment
    from .Scanner import Scanner as lexer # my superclass uses this to instantiate my scanner


    # interface
    def parse(self, uri, stream, locator):
        """
        Harvest the configuration events in {stream}
        """
        # initialize my context
        self.name = []
        self.family = []
        # set up my processor
        processor = self.processor(locator)
        # tokenize the {stream}
        self.scanner.pyre_tokenize(uri=uri, stream=stream, client=processor)
        # all done
        return self.configuration


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize the list of harvested configuration events
        self.configuration = []
        # the production table
        self.productions = {
            # the ignorables
            self.scanner.start: self.ignore,
            self.scanner.comment: self.ignore,
            self.scanner.whitespace: self.ignore,
            self.scanner.finish: self.ignore,

            # context specifier
            self.scanner.secbeg: self.context,
            # assignment
            self.scanner.key: self.assignment
            }
        # and the list of errors encountered during parsing
        self.errors = []
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
            yield from production(current=token)
        # all done
        return


    def ignore(self, **kwds):
        """
        Do nothing
        """
        # there is nothing to do here
        return []


    def context(self, current):
        """
        Process a section fragment and use it to specify the assignment context
        """
        # current is guaranteed to be a '['; get the next one
        current = yield
        # if it is not a {key}
        if type(current) is not self.scanner.key:
            # we have an error
            msg = "expected an identifier; got {.lexeme!r}".format(current)
            # handle it
            self.handleError(description=msg, locator=current.locator)
            # not much more to do
            return
        # set the family to the lexeme
        self.family = current.lexeme.split('.')

        # get the next token
        current = yield

        # if it is a {secend} marker
        if type(current) is self.lexer.secend:
            # clear the component name
            self.name = ()
            # and we are done
            return

        # if it is not a name marker
        if type(current) is not self.scanner.marker:
            # we have an error
            msg = "expected a '#'; got {.lexeme!r}".format(current)
            # handle it
            self.handleError(description=msg, locator=current.locator)
            # and bail
            return

        # get the next token
        current = yield
        # if it is not a key
        if type(current) is not self.scanner.key:
            # we have an error
            msg = "expected an identifier; got {.lexeme!r}".format(current)
            # handle it
            self.handleError(description=msg, locator=current.locator)
            # and bail
            return

        # set the name to the lexeme
        self.name = current.lexeme.split('.')

        # get the next token
        current = yield
        # if it is not a {secend} marker
        if type(current) is not self.lexer.secend:
            # we have an error
            msg = "expected a ']'; got {.lexeme!r}".format(current)
            # handle it
            self.handleError(description=msg, locator=current.locator)
            # and bail
            return

        # all done
        return


    def assignment(self, current):
        """
        Process a key assignment
        """
        # get the key
        key = current.lexeme.split('.')
        # save its locator
        locator = current.locator
        # grab the next token
        current = yield
        # if it is a value
        if type(current) is self.scanner.value:
            # extract it
            value = current.lexeme.strip()
        # otherwise
        else:
            # push it back
            self.scanner.pyre_pushback(current)
            # build an empty value
            value = None

        # time to build an assignment; if the component name is not empty
        if self.name:
            # build a conditional assignment
            event = self.ConditionalAssignment(
                component = self.name + key[:-1],
                condition = (self.name, self.family),
                key = key[-1:], value = value,
                locator = locator)
        # otherwise
        else:
            # build an unconditional assignment
            event = self.Assignment(key = self.family + key, value = value, locator = locator)
        # in any case, add it to the pile
        self.configuration.append(event)
        # and return
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


    # private data
    name = () # context: the current component name
    family = () # context: the current component family
    productions = None # the table of token handlers
    configuration = None # the list of configuration events harvested from the input source


# end of file
