# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Typed:
    """
    A class decorator that embeds type decorated subclasses whose names match their {typename}
    """


    # the list of types that i will use to decorate my client
    from . import schemata, numeric, sequences, mappings


    # meta-methods
    def __new__(cls, record=None, schemata=schemata, **kwds):
        """
        Trap use without an argument list and do the right thing
        """
        # if used without an argument, i get {record} at construction time
        if record is not None:
            # decorate it and return it; because the type returned doesn't match my type, the
            # constructor does not get invoked
            return cls.build(client=record)
        # otherwise, do the normal thing
        return super().__new__(cls, **kwds)


    def __init__(self, schemata=schemata, **kwds):
        """
        Build an instance of this decorator.
        """
        # chain up
        super().__init__(**kwds)
        # save my schemata
        self.schemata = schemata
        # all done
        return


    def __call__(self, client):
        """
        Build a class record
        """
        # delegate to my implementation
        return self.build(client=client, schemata=self.schemata)


    # implementation details
    @classmethod
    def build(cls, client, schemata=schemata):
        """
        Embed within {client} subclasses of its direct ancestor that also derive from the types in
        my {schemata}
        """
        # we do this in two passes over the tuple of schemata: once to get the pedigree of each
        # typed class, and once again to build the actual class record. this avoids the problem
        # of having the client record modified while we are hunting for custom mixins

        # temporary storage for the ancestors of each schema
        pedigree = []
        # make a pass collecting all the ancestors for each schema, BEFORE we make any
        # modifications to the client
        for schema in schemata:
            # get the ancestors for this schema
            pedigree.append(tuple(cls.pedigree(client, schema)))

        # once again, to build the classes
        for schema, ancestors in zip(schemata, pedigree):
            # make a docstring
            doc = "A subclass of {!r} of type {!r}".format(client.__name__, schema.typename)
            # build the class: name it after the schema, add the docstring
            typedClient = type(schema.typename, ancestors, {"__doc__": doc})
            # and attach it to the client
            setattr(client, schema.typename, typedClient)

        # return the new class record
        return client


    @classmethod
    def pedigree(cls, client, schema):
        """
        Build the ancestry of the client
        """
        # get the name of the type we are building
        typename = schema.typename

        # if the client declares a mixin for this type
        try:
            # use it
            yield getattr(client, typename)
        # if not
        except AttributeError:
            # no worries
            pass

        # handle the numeric types
        if schema in cls.numeric:
            # check whether the client has a 'numeric' mixin
            try:
                # and use it
                yield client.numeric
            # if not
            except AttributeError:
                # no worries
                pass

        # handle the sequences
        if schema in cls.sequences:
            # check whether the client has a 'sequence' mixin
            try:
                # and use it
                yield client.sequences
            # if not
            except AttributeError:
                # no worries
                pass

        # handle the sequences
        if schema in cls.mappings:
            # check whether the client has a 'sequence' mixin
            try:
                # and use it
                yield client.mappings
            # if not
            except AttributeError:
                # no worries
                pass

        # check whether the client provides a custom base class
        try:
            # and use it
            yield client.schema
        # if not
        except AttributeError:
            # no worries
            pass

        # now, the client
        yield client

        # finally, the {schema} itself
        yield schema

        # all done
        return


# end of file
