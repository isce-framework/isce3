<?xml version="1.0" encoding="utf-8"?>
<!--
!
! michael a.g. aïvázis
! orthologue
! (c) 1998-2019 all rights reserved
!
-->

<config>

  <!-- global settings -->
  <bind property="sample.file">sample.pml</bind>

  <!-- data for component_class_configuration -->
  <component family="sample.configuration">
    <!-- some bindings -->
    <bind property="p1">sample - p1</bind>
    <bind property="p2">sample - p2</bind>
  </component>

  <!-- data for component_class_configuration_inheritance -->
  <component family="sample.base">
    <!-- some bindings -->
    <bind property="common">base - common</bind>
  </component>

  <component family="sample.a2">
    <!-- some bindings -->
    <bind property="common">a2 - common</bind>
  </component>

  <component family="sample.derived">
    <!-- some bindings -->
    <bind property="extra">derived - extra</bind>
    <bind property="common">derived - common</bind>
  </component>

  <!-- data for component_class_configuration_inheritance_multi -->
  <component family="sample.base.multi">
    <!-- some bindings -->
    <bind property="common">base - common</bind>
  </component>

  <component family="sample.a1.multi">
    <!-- some bindings -->
    <bind property="common">a1 - common</bind>
    <bind property="middle">a1 - middle</bind>
  </component>

  <component family="sample.a2.multi">
    <!-- some bindings -->
    <bind property="common">a2 - common</bind>
  </component>

  <component family="sample.derived.multi">
    <!-- some bindings -->
    <bind property="extra">derived - extra</bind>
  </component>

  <!-- data for component_class_inventory -->
  <component family="sample.inventory.base">
    <!-- some bindings -->
    <bind property="b">1</bind>
  </component>

  <component family="sample.inventory.derived">
    <!-- some bindings -->
    <bind property="b">2</bind>
    <bind property="d">Hello world!</bind>
  </component>

  <!-- data for component_instance_configuration -->
  <component name="c" family="sample.configuration">
    <!-- some bindings -->
    <bind property="p1">p1 - instance</bind>
    <bind property="p2">p2 - instance</bind>
  </component>

  <!-- data for component_instance_configuration_inheritance -->
  <component name="d" family="sample.derived">
    <!-- some bindings -->
    <bind property="extra">d - extra</bind>
    <bind property="middle">d - middle</bind>
    <bind property="common">d - common</bind>
  </component>

  <!-- data for component_instance_configuration_inheritance_multi -->
  <component name="d" family="sample.derived.multi">
    <!-- some bindings -->
    <bind property="extra">d - extra</bind>
  </component>

  <!-- data for component_instance_binding_configuration -->
  <component name="c" family="sample.manager">
    <bind property="jobs">10</bind>
  </component>
  <component name="w" family="sample.worker">
    <bind property="host">pyre.orthologue.com</bind>
  </component>
  <component name="c.gopher" family="sample.worker">
    <bind property="host">foxtrot.orthologue.com</bind>
  </component>

  <!-- data for component_aliases -->
  <component family="sample.functor">
    <bind property="mean">mean</bind>
    <bind property="μ">μ</bind>
    <bind property="spread">spread</bind>
    <bind property="σ">σ</bind>
  </component>

  <!-- data for component_catalog -->
  <component name="catalog_container" family="sample.container">
    <!-- put some components in the catalog -->
    <component name="catalog">
      <bind property="cat1">sample.ifac.comp</bind>
      <bind property="cat2">sample.ifac.comp#foo</bind>
      <bind property="cat3">sample.ifac.comp</bind>
    </component>
  </component>

  <component name="catalog_container.catalog.cat1" family="sample.ifac.comp">
      <bind property="tag">cat1</bind>
  </component>

  <component name="catalog_container.catalog.cat2" family="sample.ifac.comp">
      <bind property="tag">cat2</bind>
  </component>

  <component name="catalog_container.catalog.cat3" family="sample.ifac.comp">
      <bind property="tag">cat3</bind>
  </component>

  <component name="foo" family="sample.ifac.comp">
      <bind property="tag">cat2</bind>
  </component>

  <!-- data for component_list -->
  <component name="alec" family="sample.person">
    <!-- put some components in the list -->
    <bind property="activities">
      study#physics, relax#wow, sleep#nap
    </bind>
  </component>

  <component name="physics" family="sample.activities.study">
    <bind property="duration">30*minute</bind>
  </component>

  <component name="wow" family="sample.activities.relax">
    <bind property="duration">60*minute</bind>
  </component>

  <component name="nap" family="sample.activities.sleep">
    <bind property="duration">3*hour</bind>
  </component>

</config>


<!-- end of file -->
