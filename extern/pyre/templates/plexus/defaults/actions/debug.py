# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# get the package
import {project.name}

# declaration
class debug({project.name}.command, family='{project.name}.actions.debug'):
    """
    An example command
    """


    # meta-data
    pyre_tip = "an example of a user supplied command"


    # behaviors
    @{project.name}.export(tip='convenience action for debugging the plexus')
    def test(self, plexus, **kwds):
        """
        Convenient resting point for debugging code during development
        """
        # show me
        plexus.info.log(self.pyre_name)
        # all done
        return


# end of file
