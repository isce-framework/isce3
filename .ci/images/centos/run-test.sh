source /opt/docker/bin/entrypoint_source \
 && cd build \
 && ctest -j `nproc` --nocompress-output --output-on-failure -T Test || true \
 && cp Testing/\$\(head -1 Testing/TAG\)/Test.xml .
