# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Symbolic values for the possible actions to perform when a foreign key is deleted or
updated. Consult the SQL standard for their meaning
"""

# produce an error indicating that the change would create foreign key constraint violation; if
# the constraint is deferred; this is the default
noAction = "NO ACTION"

# produce an error indicating that the change would create foreign key constraint violation
restrict = "RESTRICT"

# on delete, delete all rows that reference the deleted row; on update, change the value of the
# referencing field to the new value of the foreign key
cascade = "CASCADE"

# set the referencing field to NULL
setNull = "SET NULL"

# set the referencing field to its default value
setDefault = "SET DEFAULT"


# end of file
