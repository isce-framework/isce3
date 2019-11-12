# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

PROJECT = pyre
PROJ_TIDY += __pycache__

all: test

test: sanity metaclasses protocols components configurations watching clean

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py

metaclasses:
	${PYTHON} ./requirement.py
	${PYTHON} ./role.py
	${PYTHON} ./actor.py

protocols:
	${PYTHON} ./protocol.py
	${PYTHON} ./protocol_family.py
	${PYTHON} ./protocol_behavior.py
	${PYTHON} ./protocol_property.py
	${PYTHON} ./protocol_inheritance.py
	${PYTHON} ./protocol_shadow.py
	${PYTHON} ./protocol_inheritance_multi.py
	${PYTHON} ./protocol_compatibility.py
	${PYTHON} ./protocol_compatibility_reports.py
	${PYTHON} ./protocol_instantiation.py

components: component-basics component-class component-instance component-multi

component-basics:
	${PYTHON} ./component.py
	${PYTHON} ./component_family.py
	${PYTHON} ./component_behavior.py
	${PYTHON} ./component_property.py
	${PYTHON} ./component_facility.py
	${PYTHON} ./component_inheritance.py
	${PYTHON} ./component_shadow.py
	${PYTHON} ./component_inheritance_multi.py
	${PYTHON} ./component_compatibility.py
	${PYTHON} ./component_compatibility_reports.py
	${PYTHON} ./component_implements.py
	${PYTHON} ./component_bad_implementations.py

component-class:
	${PYTHON} ./component_class_registration.py
	${PYTHON} ./component_class_registration_inventory.py
	${PYTHON} ./component_class_registration_model.py
	${PYTHON} ./component_class_configuration.py
	${PYTHON} ./component_class_configuration_inheritance.py
	${PYTHON} ./component_class_configuration_inheritance_multi.py
	${PYTHON} ./component_class_binding.py
	${PYTHON} ./component_class_binding_import.py
	${PYTHON} ./component_class_binding_vfs.py
	${PYTHON} ./component_class_binding_conversions.py
	${PYTHON} ./component_class_validation.py
	${PYTHON} ./component_class_trait_matrix.py
	${PYTHON} ./component_class_private_locators.py
	${PYTHON} ./component_class_public_locators.py
	${PYTHON} ./component_class_inventory.py

component-instance:
	${PYTHON} ./component_defaults.py
	${PYTHON} ./component_instantiation.py
	${PYTHON} ./component_invocation.py
	${PYTHON} ./component_instance_registration.py
	${PYTHON} ./component_instance_configuration.py
	${PYTHON} ./component_instance_configuration_constructor.py
	${PYTHON} ./component_instance_configuration_inheritance.py
	${PYTHON} ./component_instance_configuration_inheritance_multi.py
	${PYTHON} ./component_instance_binding.py
	${PYTHON} ./component_instance_binding_implicit.py
	${PYTHON} ./component_instance_binding_configuration.py
	${PYTHON} ./component_instance_binding_existing.py
	${PYTHON} ./component_instance_binding_deferred.py
	${PYTHON} ./component_instance_validation.py
	${PYTHON} ./component_instance_private_locators.py
	${PYTHON} ./component_instance_public_locators.py
	${PYTHON} ./component_aliases.py --functor.μ=0.10 --gaussian.σ=0.10

component-multi:
	${PYTHON} ./component_slots.py
	${PYTHON} ./component_list.py
	${PYTHON} ./component_dict.py

configurations:
	${PYTHON} ./quad.py

watching:
	${PYTHON} ./monitor.py
	${PYTHON} ./tracker.py


# end of file
