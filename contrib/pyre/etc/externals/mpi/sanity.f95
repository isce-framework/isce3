! -*- F90 -*-
!
! michael a.g. aïvázis
! orthologue
! (c) 1998-2019 all rights reserved
!

program sanity

    ! initialize the status flag
    integer status
    ! initialize MPI
    call MPI_Init(status)
    ! finalize MPI
    call MPI_Finalize(status)

    end

! end of file
