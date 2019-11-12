# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from ..patterns import coroutine
# my superclass
from .Lexer import Lexer


# class declaration
class Scanner(metaclass=Lexer):
    """
    The input stream tokenizer
    """


    # types
    # exceptions
    from .exceptions import ParsingError, TokenizationError
    # the descriptor factory
    from .Descriptor import Descriptor as pyre_token
    # the stream wrapper
    from .InputStream import InputStream as pyre_inputStream
    # the default tokens; all scanners have these
    start = pyre_token()
    finish = pyre_token()
    whitespace = pyre_token(pattern=r'\s+')


    # interface
    def pyre_tokenize(self, uri, stream, client):
        """
        Extract lines from {stream}, convert them into token streams and send them to {client}
        """
        # show me
        # print(' ++ pyre.parsing.Scanner:')
        # print('      uri={}'.format(uri))
        # print('      stream={}'.format(stream))
        # print('      client={}'.format(client))

        # save the source information
        self.pyre_client = client
        # build and save the input stream
        stream = self.pyre_stream = self.pyre_inputStream(uri=uri, stream=stream)
        # my tokenizer
        tokenizer = self.pyre_tokenizer

        # get the token cache
        cache = self.pyre_cache = []
        # to get things going, build a {start} token and pass it along to the {client}
        self.pyre_start()

        # nothing wrong has happened
        recovering = False
        # until something special happens
        while True:
            # get whatever tokens have accumulated in the cache
            for token in cache:
                # and send them off
                client.send(token)
            # flush the cache
            cache.clear()
            # try to
            try:
                # to get a token match
                match = stream.match(scanner=self, tokenizer=tokenizer)
            # if the stream ran out of text
            except StopIteration:
                # wrap up by sending a {finish} token to the client
                self.pyre_finish()
                # and terminate the loop
                break
            # if a tokenization error occurred
            except self.TokenizationError as error:
                # if this is the first time this
                if not recovering:
                    # invoke the downstream error handler
                    client.throw(self.TokenizationError, error)
                    # go into error recovery mode
                    recovering = True
                # skip the current character
                stream.column += 1
                # and try again
                continue

            # we have match; clear the fault indicator
            recovering = False
            # lookup the name of the token
            name = match.lastgroup
            # get the token class
            factory = getattr(self, name)
            # make a token
            token = factory(lexeme = match.group(name), locator = self.pyre_stream.locator)
            # show me
            # print(token)
            # process it
            client.send(token)

        # all done
        return


    def pyre_start(self):
        """
        Indicate the beginning of scanning
        """
        # to get things going, build a {start} token
        start = self.start(locator=self.pyre_stream.locator)
        # and send it along
        self.pyre_client.send(start)
        # all done
        return


    def pyre_finish(self):
        """
        Indicate that scanning is complete
        """
        # to wrap things up, build a {finish} token
        finish = self.finish(locator=self.pyre_stream.locator)
        # and send it along
        self.pyre_client.send(finish)
        # reset my state
        self.pyre_stream = None
        self.pyre_client = None
        # all done
        return


    def pyre_pushback(self, token):
        """
        Push a token back into the token stream
        """
        # do it
        self.pyre_cache.append(token)
        # all done
        return self


    def pyre_newline(self, stream):
        """
        Hook invoked when a new line of text is pulled from the input stream
        """
        # nothing to do by default
        return


    # helpers
    @coroutine
    def pyre_ignoreWhitespace(self, client):
        """
        Remove {whitespace} tokens from the input stream
        """
        # support for upstream error notification
        fault = None
        # for ever
        while True:
            # attempt to
            try:
                # get a token
                token = yield fault
            # if anything goes wrong
            except self.ParsingError as error:
                # forward it to my client
                client.throw(type(error), error)
            # if it is not whitespace
            if not isinstance(token, self.whitespace):
                # pass it along
                fault = client.send(token)
        # all done
        return


    # implementation details
    # set by my meta-class
    pyre_tokens = None # a list of my tokens
    pyre_tokenizer = None # the compiled regex constructed out the patterns of my tokens

    # tokenizing state
    pyre_stream = None
    pyre_client = None
    pyre_cache = None # the list of tokens that have been pushed back


# end of file
