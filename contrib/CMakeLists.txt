add_subdirectory(googletest)

# Copy cereal to installation include path
set(CEREAL_INCLUDE cereal)
install(DIRECTORY ${CEREAL_INCLUDE} DESTINATION include COMPONENT cereal_headers)
