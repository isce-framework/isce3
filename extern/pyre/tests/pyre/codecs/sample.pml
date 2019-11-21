<?xml version="1.0" encoding="utf-8"?>
<!--
!
! michael a.g. aïvázis
! orthologue
! (c) 1998-2019 all rights reserved
!
-->

<config>

  <package name="gauss">
    <bind property="name">gauss</bind>

    <component name="mc">
      <bind property="samples">10**6</bind>
      <bind property="integrand">import:gauss.fuctors.gaussian</bind>

      <component name="box">
        <bind property="diagonal">((-1,-1), (1,1))</bind>
      </component>

      <component name="integrand" family="gauss.functors.gaussian">
        <bind property="μ">(0,0)</bind>
        <bind property="σ">1/3</bind>
      </component>

    </component>

  </package>
</config>


<!-- end of file -->
