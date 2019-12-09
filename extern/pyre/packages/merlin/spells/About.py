# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import merlin


# declaration
class About(merlin.spell, family='merlin.spells.about'):
    """
    Display information about this application
    """


    # user configurable state
    prefix = merlin.properties.str()
    prefix.tip = "specify the portion of the namespace to display"


    # class interface
    @merlin.export(tip="print the copyright note")
    def copyright(self, plexus, **kwds):
        """
        Print the copyright note of the merlin package
        """
        # make some space
        plexus.info.line()
        # print the copyright note
        plexus.info.log(merlin.meta.header)
        # all done
        return


    @merlin.export(tip="print out the acknowledgments")
    def credits(self, plexus, **kwds):
        """
        Print the acknowledgments
        """
        # make some space
        plexus.info.line()
        # print the acknowledgments
        plexus.info.log(merlin.meta.acknowledgments)
        # all done
        return


    @merlin.export(tip="print out the license and terms of use")
    def license(self, plexus, **kwds):
        """
        Print the license and terms of use of the merlin package
        """
        # make some space
        plexus.info.line()
        # print the license
        plexus.info.log(merlin.meta.license)
        # all done
        return


    @merlin.export(tip='dump the application configuration namespace')
    def nfs(self, plexus, **kwds):
        """
        Dump the application configuration namespace
        """
        # get the prefix
        prefix = self.prefix or 'merlin'
        # show me
        plexus.pyre_nameserver.dump(prefix)
        # all done
        return


    @merlin.export(tip="print the version number")
    def version(self, plexus, **kwds):
        """
        Print the version of the merlin package
        """
        # print the version number as simply as possible
        print(merlin.meta.version)
        # all done
        return


    @merlin.export(tip='dump the application virtual filesystem')
    def vfs(self, plexus, **kwds):
        """
        Dump the application virtual filesystem
        """
        # get the prefix
        prefix = self.prefix or '/merlin'
        # build the report
        report = '\n'.join(plexus.vfs[prefix].dump())
        # sign in
        plexus.info.line('vfs: prefix={!r}'.format(prefix))
        # dump
        plexus.info.log(report)
        # all done
        return


# end of file
