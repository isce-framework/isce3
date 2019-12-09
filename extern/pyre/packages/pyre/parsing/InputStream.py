# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import io
import pyre.tracking


# declaration
class InputStream:
    """
    A wrapper over input streams that maintains location information
    """


    # types
    from .exceptions import TokenizationError

    # public data
    uri = None
    stream = None
    line = 0
    column = 0
    text = None

    @property
    def locator(self):
        """
        Build and return a locator to my current position in my input stream
        """
        # build a file locator
        marker = pyre.tracking.file(
            source=self.uri, line=self.line, column=self.column)
        # and return it
        return marker


    # interface
    def match(self, scanner, tokenizer):
        """
        Attempt to match the text at my current position using the given regular expression
        """
        # check whether i need to consume more input
        self.update(client=scanner)
        # attempt a match
        match = tokenizer.match(self.text, pos=self.column)
        # if there is no match
        if not match:
            # build an error descriptor
            fault = self.TokenizationError(text=self.text[self.column:], locator=self.locator)
            # and complain
            raise fault

        # otherwise, update my column
        self.column = match.end()
        # and return the match
        return match


    # meta-methods
    def __init__(self, uri, stream, line=0, column=0, **kwds):
        # chain up
        super().__init__(**kwds)
        # set the uri
        self.uri = uri
        # initialize my position
        self.line = line
        self.column = column
        self.text = ''
        # if the stream is not open in text mode
        if not isinstance(stream, io.TextIOBase):
            # wrap it
            stream = io.TextIOWrapper(stream)
        # and save it
        self.stream = stream
        # all done
        return


    # implementation details
    def update(self, client=None):
        """
        Prepare to process more text from my input stream
        """
        # if i don't need to grab another line
        if self.column < len(self.text):
            # all done
            return self
        # otherwise, adjust my state
        self.line += 1
        self.column = 0
        # attempt to
        try:
            # get a line
            self.text = next(self.stream)
        # if the stream is exhausted
        except StopIteration:
            # treat as a blank line
            self.text = ''
            # and let the client know
            raise

        # if someone is interested that I pulled a new line from the stream
        if client:
            # let him know
            client.pyre_newline(stream=self)

        # all done
        return self


# end of file
