<?xml version="1.0" encoding="utf-8"?>
<!--
!
! michael a.g. aïvázis
! orthologue
! (c) 1998-2019 all rights reserved
!
-->

<config>

  <!-- some global settings -->
  <component family="pyre.db.server.sqlite">
    <bind property="database">:memory:</bind>
  </component>

  <!--  settings for particular test cases -->
  <component name="test" family="pyre.db.server.sqlite">
    <bind property="database">pyre.sql</bind>
  </component>

</config>


<!-- end of file -->
