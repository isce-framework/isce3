# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package contains the implementation of the pml reader and writer

A well formed pml document is an XML document that contains the following tags:

  <config>: the top level tag that corresponds to the entire document
  <package>: establishes a namespace for the bindings it contains.
  <component>: introduces the configuration section for a component
  <bind>: establishes the value of a property

As a simple example,

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <bind property="pyre.version">1.0</bind>
   </config>

assign the value "1.0" to the property "pyre.version". This is equivalent to supplying

   --pyre.version=1.0

on the command line. If there are many properties in the pyre namespace, you can group them
together inside a package tag, so you don't have to repeat the "pyre" part of the name.

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <package name="pyre">
           <bind property="version">1.0</bind>
       </package>
   </config>

These two pml documents have the same net effect on the configuration store. This equivalence
is established by treating the character '.' in the names of properties and packages as a
namespace level separator, similar to the way it is treated on the command line.  Package tags
can nest arbitrarily deeply, with each level adding further qualifications to its parent
namespace.

A more common use of pml files is to store component configuration settings. Such settings use
<bind> tags nested inside <component> tags, and take various forms. To begin with, you can
configure the defaults that are applied to all instances of a given component class. For
example, consider the package {gauss} from the examples directory. It contains a sub-package
{functors} with representations of a few functions that help illustrate the kind of flexibility
provided by pyre. If you look at the component {Constant} in this package, you will find it has
a single property named {value}, with a default value of 1. To override this default and set it
to 0 you would create a file {gauss.pml} with the following contents:

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component family="gauss.functors.constant">
           <bind property="value">0</bind>
       </component>
   </config>

The framework knows that you are trying to establish a new default for ALL instances of the
functor {Constant} because the <component> tag only specifies the attribute {family} with a
value that matches the family attribute in the {Constant} declaration. If you found yourself
wanting to redefine the default values of many of the components in the {gauss} package, you
could restructure this file as follows:

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <package name="gauss">
           <component family="functors.constant">
               <bind property="value">0</bind>
           </component>
       </package>
   </config>

Every tag that is nested inside a <package} tag is placed automatically within the specified
namespace, saving you from having to keep mentioning the package name.

You may remember that pyre component instances are required to have a unique name. Suppose that
you have created an instance of {Constant} using the convenience factory method exported by the
{gauss} package

    import gauss
    π = gauss.functors.constant(name='π')

If you ask this instance for its {value} you will get 1, the default specified in the component
declaration. You can override this default for components named {π} by placing

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="π">
           <bind property="value">3.1415926</bind>
       </component>
   </config>

in a configuration file. You should be aware that component names live in the same namespace as
everything else in the framework, so it is important to establish a naming scheme for your
application components and avoid name collisions.

The above configuration file will force the framework to attempt to set a property named
{value} for every component named {π}, regardless of whether it has such a property or not. In
order to make sure that only {Constant} instances are subject to this configuration, you can
use the third form of the <component> tag:

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="π" family="gauss.functors.constant">
           <bind property="value">3.1415926</bind>
       </component>
   </config>

which explicitly specifies both the name and the family of components that should be sensitive
to this configuration setting.

Many components use other components to carry out part of their responsibilities. For example,
the Monte Carlo integrator in the {gauss} package uses four separate components to completely
specify the integral to perform. Let's try to configure an instance named {mc} to use 10^6
samples to integrate the functor {Gaussian}, with all other settings left at their defaults:

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="mc">
           <bind property="samples">10**6</bind>
           <bind property="integrand">import:gauss.functors.gaussian</bind>
       </component>
   </config>

We can change the region of integration to the square from (-1,-1) to (1,1) by nesting a
component configuration

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="mc">
           <bind property="samples">10**6</bind>
           <bind property="integrand">import:gauss.functors.gaussian</bind>

           <component name="box">
               <bind property="diagonal">((-1,-1), (1,1))</bind>
           </component>

       </component>
   </config>

At this point, you might be tempted to add some configuration settings for the integrand by
doing something like

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="mc">
           <bind property="samples">10**6</bind>
           <bind property="integrand">import:gauss.functors.gaussian</bind>

           <component name="box">
               <bind property="diagonal">((-1,-1), (1,1))</bind>
           </component>

           <component name="integrand">
               <bind property="μ">(0,0)</bind>
               <bind property="σ">1/3</bind>
           </component>
       </component>
   </config>

There is a problem with this approach, however: suppose that you later override your choice of
integrand by specifying a different functor on the command line. The framework will still
attempt to set the attributes {μ} and {σ} to the values specified above, almost certainly not
what you expected. The solution is once again to specify the type of the component along its
name so that configuration settings are applied only when appropriate.

   <?xml version="1.0" encoding="utf-8"?>
   <config>
       <component name="mc">
           <bind property="samples">10**6</bind>
           <bind property="integrand">import:gauss.functors.gaussian</bind>

           <component name="box">
               <bind property="diagonal">((-1,-1), (1,1))</bind>
           </component>

           <component name="integrand" family="gauss.functors.gaussian">
               <bind property="μ">(0,0)</bind>
               <bind property="σ">1/3</bind>
           </component>
       </component>
   </config>

"""


# pull the codec
from .PML import PML as pml


# end of file
