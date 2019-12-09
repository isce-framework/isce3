# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
import itertools
from .. import tracking


# class declaration
class CommandLineParser:
    """
    Support for parsing the application command line

    The general form of a command line configuration event is
        --key=value
    which creates a configuration event that will bind {key} to {value}.

    This implementation supports the following constructs:
        --key
            set key to None
        --key=value
        --key.subkey=value
            key may have an arbitrary number of period delimited levels
        --(key1,key2)=value
            equivalent to --key1=value and --key2=value; an arbitrary number of comma separated
            key names are allowed
        --key.(key1,key2)=value
        --key.(key1,key2).key3=value
            as above; this form is supported at any level of the hierarchy

    By default, instances of the command line parser use the following literals
        '-': introduces a configuration command
        '=': indicates an assignment
        '.': the separator for multi-level keys
        '(' and ')': the start and end of a key group
        ',': the separator for keys in a group

    If you want to change any of this, you can instantiate a command line parser, modify any of
    its public data, and invoke "buildScanners" to recompute the associated regular expression
    engines
    """


    # public data
    prefix = '-'
    assignment = '='
    fieldSeparator = '.'
    groupStart = '('
    groupSeparator = ','
    groupEnd = ')'

    handlers = None # the special handlers of command line arguments
    assignmentScanner = None
    locator = staticmethod(tracking.command)


    # types
    from .events import Command, Assignment


    # interface
    def parse(self, argv):
        """
        Harvest the configuration events in {argv} and store them in a {configuration}

        parameters:
            {argv}: a container of strings of the form "--key=value"
            {locator}: an optional locator; not used by this decoder
        """
        # build a configuration object to store the processed command line
        configuration = []
        # run through the command line
        for index,arg in enumerate(argv):
            # look for an assignment
            match = self.assignmentScanner.match(arg)
            # if we have one
            if match:
                # get the tokens from the scanner
                key = match.group("key")
                value = match.group("value") or ''
                # if there is a key
                if key:
                    # process this
                    self._processAssignments(
                        configuration, key,value, self.locator(arg=match.string))
                # if not, something special happened
                else:
                    # we ran in to a '-' or '--' that signals the end of configuration options
                    index += 1
                    # record the rest of the command line
                    self._processArguments(configuration, index, *argv[index:])
                    # not our problem any more
                    break
            # else it must be a regular command line argument
            else:
                # record it
                self._processArguments(configuration, index, arg)
        # all done; return the configuration
        return configuration


    def buildScanners(self):
        """
        Build the command line recognizers that are used to detect the supported command line
        argument syntactical forms
        """
        # the assignment recognizer regular expression
        regex = []
        # if i have a special character that indicates the beginning of a cofiguration
        if self.prefix:
            # incorporate it into the regex
            regex.append(r'(?P<prefix>' + self.prefix + r'{1,2})')
        # add the 'key=value' form
        regex += [
            # the key
            r'(?P<key>[^', self.assignment, r']*)',
            # the optional assignment symbol
            self.assignment, r'?',
            # and the optional value
            r'(?P<value>.+)?'
            ]
        # compile this pattern
        self.assignmentScanner = re.compile("".join(regex))
        # all done
        return


    # meta methods
    def __init__(self, handlers=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # build the scanners
        self.buildScanners()
        # the list of registered handlers of command line events
        self.handlers = {} if handlers is None else handlers
        # all done
        return


    # implementation details
    def _processAssignments(self, configuration, key, value, locator):
        """
        Handler for command line arguments that were interpreted as assignments

        Look for the supported shorthands and unfold them into canonical forms.
        """
        # reset the pile of parts
        fields = []

        # split the key on the field separator to identify the various fields
        for field in key.split(self.fieldSeparator):
            # check for field distribution
            if field[0] == self.groupStart and field[-1] == self.groupEnd:
                # got one; split on the group separator
                fields.append(field[1:-1].split(self.groupSeparator))
            # otherwise
            else:
                # just store the field name
                fields.append([field])

        # now, form all the specified addresses by computing the Cartesian product
        for spec in itertools.product(*fields):
            # check whether
            try:
                # there is a handler registered for this spec
                handler = self.handlers[spec[0]]
            # nope, not there
            except KeyError:
                # create a new assignment
                event = self.Assignment(key=spec, value=value, locator=locator)
                # add it to the pile
                configuration.append(event)
            # if there is
            else:
                # invoke it
                handler(key=spec, value=value, locator=locator)

        # all done
        return


    def _processArguments(self, configuration, index, *args):
        """
        Interpret {args} as application commands and store them in {configuration}
        """
        # iterate over the command line arguments that were handed to me
        for arg in args:
            # build a command request
            event = self.Command(command=arg, locator=self.locator(arg=arg))
            # add it to the pile
            configuration.append(event)
            # update the index
            index += 1
        # all done
        return


# end of file
