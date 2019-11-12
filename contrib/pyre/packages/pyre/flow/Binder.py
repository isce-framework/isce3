# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre


# declaration
class Binder:
    """
    Method decorator that constructs {operator} nodes to connect {inputs} to {outputs}
    """


    # meta-methods
    def __new__(cls, method=None, inputs=None, outputs=None, **kwds):
        """
        Trap the invocation with meta-data and delay the decoration of the method
        """
        # show me
        print(f"Binder.__new__:")
        print(f"    method: {method}")
        print(f"    inputs: {inputs}")
        print(f"    outputs: {outputs}")
        print(f"    kwds: {kwds}")

        # if the method is known, we were called by {python} after the method declaration
        if method is not None:
            # chain up to invoke the descriptor construction; swallow my arguments, for now;
            # they will show up again in {__init__}
            return super().__new__(cls, **kwds)

        # if we don't know the method, we were invoked before the method declaration was
        #  processed; the strategy here is to return a {Binder} constructor as the value of
        #  this invocation, which accomplishes two things: it gives python something to call
        #  when the method declaration is done, and prevents my {__init__} from getting invoked
        #  prematurely

        # here is the constructor closure
        def build(method):
            """
            Convert a component method into a binder
            """
            # just build one of my instances
            return cls(method=method, inputs=inputs, outputs=outputs, **kwds)

        # to hand over
        return build


    def __init__(self, method, inputs=None, outputs=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # show me
        print(f"Binder.__init__:")
        print(f"    method: {method}")
        print(f"    inputs: {inputs}")
        print(f"    outputs: {outputs}")
        print(f"    kwds: {kwds}")

        # save
        self.method = method
        self.inputs = inputs
        self.outputs = outputs

        # all done
        return


    def __get__(self, instance, cls):
        """
        Access to the method
        """
        # delegate
        return self.method.__get__(instance, cls)


    # implementation details
    # private data
    method = None
    inputs = None
    outputs = None


# end of file
