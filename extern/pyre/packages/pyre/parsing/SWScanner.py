# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
# superclass
from .Scanner import Scanner


# declaration
class SWScanner(Scanner):
    """
    A scanner for languages that use leading whitespace to indicate the hierarchical structure
    of the content
    """


    # exceptions
    from .exceptions import IndentationError
    # tokens
    pop = Scanner.pyre_token()
    push = Scanner.pyre_token()
    cdata = Scanner.pyre_token()
    comment = Scanner.pyre_token(head=r'(?<!\\)#', pattern='.*', tail='$')


    # scanning event handlers
    def pyre_start(self):
        """
        Scanning has begun
        """
        # initialize my current indentation position
        self.pyre_dent = 0
        # initialize the indentation stack
        self.pyre_blocks = []
        # and chain up
        return super().pyre_start()


    def pyre_finish(self):
        """
        Scanning has ended
        """
        # get my client
        client = self.pyre_client
        # and the open blocks
        blocks = self.pyre_blocks
        # make an end-of-block token
        token = self.pop(locator=self.pyre_stream.locator)
        # send a token for each open block
        for _ in blocks: client.send(token)
        # reset my indentation level
        self.pyre_dent = 0
        # clear the block stack
        blocks.clear()
        # all done here; chain up for the rest of the clean up
        return super().pyre_finish()


    # implementation details
    def pyre_newline(self, stream):
        """
        A fresh line has been retrieved from the input {stream}
        """
        # get the input line
        text = stream.text
        # if it is a trivial line
        if not self.pyre_trim(text):
            # nothing more to do with it
            return

        # otherwise, figure out its indentation level
        new = self.margin.match(text).end()
        # and get my current level
        current = self.pyre_dent

        # if we are at the same level
        if new == current:
            # nothing more to do
            return
        # if this is the start of a new block
        if new > current:
            # indent
            tokens = self.pyre_indent(locator=stream.locator, column=new)
        # otherwise
        else:
            # we are closing blocks
            tokens = self.pyre_dedent(locator=stream.locator, column=new)

        # get the client
        client = self.pyre_client
        # grab the tokens
        for token in tokens:
            # and send them to the client
            client.send(token)

        # all done
        return


    def pyre_indent(self, locator, column):
        """
        Indent to the given {column}
        """
        # make a token to mark the beginning of a block
        token = self.push(locator=locator)
        # save the current level in the stack of open blocks
        self.pyre_blocks.append(self.pyre_dent)
        # push in to the given column
        self.pyre_dent = column
        # send the token
        yield token
        # all done
        return


    def pyre_dedent(self, locator, column):
        """
        Dedent by as many open blocks as it takes to come back to {column}
        """
        # make a token to mark the end of a block
        token = self.pop(locator=locator)
        # get the stack of open blocks
        blocks = self.pyre_blocks

        # get the current dent
        dent = self.pyre_dent
        # until we run into the right block
        while column < dent:
            # mark the block as closed
            yield token
            # update the current dent by popping an open block
            dent = blocks.pop()

        # check that this brought us back to a consistent indentation level
        if dent != column:
            # and if not, build a description of the problem
            fault = self.IndentationError(text='', locator=locator)
            # and invoke the downstream error handler
            self.pyre_client.throw(self.TokenizationError, fault)

        # push out to the right spot
        self.pyre_dent = column
        # all done
        return


    def pyre_cdata(self):
        """
        Scan forward from the current location and convert any text in all nested blocks into a
        {cdata} token. The lexeme of a {cdata} token is a list of strings with the leading
        indentation, all trivial lines, and all embedded comments removed.
        """
        # get my input stream
        stream = self.pyre_stream
        # initialize the token payload
        data = []
        # make the token
        cdata = self.cdata(lexeme=data, locator=stream.locator)
        # get the rest of the current line
        rest = self.pyre_trim(stream.text[stream.column:])
        # update the cursor
        stream.column = len(stream.text)

        # if it is not a trivial line
        if rest:
            # append to the pile
            data.append(rest)
            # add the cdata token
            self.pyre_pushback(cdata)
            # all done
            return cdata

        # get the current indentation level
        dent = self.pyre_dent
        # if the block is not in line, look deeper until something interesting happens
        while True:
            # attempt to
            try:
                # get the next line
                stream.update()
            # if we ran out of input
            except StopIteration:
                # place the token in the cache
                self.pyre_pushback(cdata)
                # and bail
                return cdata

            # get the line
            text = stream.text
            # trim the line
            trimmed = self.pyre_trim(text)
            # if nothing interesting is left, move on
            if not trimmed:
                # mark this line as completely processed
                stream.column = len(text)
                # and go on
                continue
            # otherwise, figure out its indentation level
            new = self.margin.match(text).end()
            # if it is not deep enough, we are done looking for the cdata block
            if new <= dent: break
            # otherwise, put the trimmed data on the payload
            data.append(trimmed)
            # mark this line as completely processed, and go on
            stream.column = len(text)

        # we have found the end of the cdata; save it
        self.pyre_pushback(cdata)
        # pop as many blocks as necessary
        for token in self.pyre_dedent(locator=stream.locator, column=new):
            # and add the tokens to the cache
            self.pyre_pushback(token)
        # all done
        return cdata


    def pyre_trim(self, text):
        """
        Determine whether {text} contains any structurally relevant information
        """
        # eliminate comments and whitespace and return what's left
        return self.comment.scanner.sub('', text).strip()


    pyre_dent = 0 # the current indentation level
    pyre_blocks = None # the stack of open blocks

    # my indentation scanner
    margin = re.compile(r' *') # only spaces; tabs are an error


# end of file
