# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class LineComments:
    """
    The line based commenting strategy
    """


    # implemented interface
    def commentBlock(self, lines):
        """
        Create a comment block out of the given {lines}
        """
        # build the leader
        leader = self.leader + self.comment
        # iterate over the {lines}
        for line in lines:
            # if the line is not empty
            if line:
                # render it
                yield leader + ' ' + line
            # otherwise
            else:
                # render just the comment marker
                yield leader

        # all done
        return


    def commentLine(self, line):
        """
        Mark {line} as a comment
        """
        # build the leader
        leader = self.leader + self.comment
        # if the line is non-empty
        if line:
            # mark it
            return leader + ' ' + line
        # otherwise, just return the comment characters
        return leader


    # private data
    comment = None


# end of file
