# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import enum


# declaration
class TaskStatus(enum.Enum):
    """
    Indicators used by tasks to describe what happened during their execution
    """

    # as far as the task can tell, it completed successfully
    completed = 0

    # the task failed to complete, but the failure is temporary and the process is in good
    # state; for example, some resource required to complete the task is currently unavailable;
    # the task should be rescheduled, and the process is ready for new work
    failed = 1

    # this task cannot be completed successfully; for example, executing the task raised
    # exceptions that indicated configuration, logic, or run-time errors; this task should be
    # removed from the workplan and the user should be notified
    aborted = 2

    # the task failed because the process is compromised and can't be relied upon any
    # more; the task should be rescheduled to another process; this worker should be removed
    # from the pool permanently
    damaged = 3


# end of file
