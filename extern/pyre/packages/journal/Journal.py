# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre

# access the requirements
from .Device import Device


# declaration
class Journal(pyre.component, family="journal.executive"):
    """
    Place holder for the configurable bits of the journal package
    """


    # class public data
    device = Device()
    device.doc = "the component responsible for recording journal entries"


    # interface
    def configureCategories(self, categories):
        """
        Extract channel information from the configuration store for each of the given
        {categories}
        """
        # access the nameserver
        ns = self.pyre_executive.nameserver
        # access the type converters
        import pyre.schemata
        # pick the one for booleans
        schema = pyre.schemata.bool()
        # and iterate over {categories}, updating their indices with the contents of the pyre
        # configuration store
        for category in categories:
            # build the key prefix
            prefix = "journal\." + category.severity
            # identify the relevant keys
            for info, node in ns.find(pattern=prefix):
                # get the value
                value = node.value
                # if it's {None}, it probably came from the command line without an assignment
                if value is None: value = True
                # attempt to cast to a bool
                try:
                    value = schema.coerce(value)
                # if this fails
                except schema.CastingError:
                    # ignore it and move on
                    continue
                # extract the category name
                categoryName = '.'.join(info.name.split('.')[2:])
                # update the index
                category(categoryName).active = value
        # all done
        return


# end of file
