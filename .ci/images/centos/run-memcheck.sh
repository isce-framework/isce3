source /opt/docker/bin/entrypoint_source \
 && cd build \
 && ctest --nocompress-output --output-on-failure -T Test || true \
 && cp Testing/\$(head -1 Testing/TAG)/Test.xml . \
 && ctest --no-compress-output --output-on-failure --timeout 10000 -T MemCheck \
        -E test.cxx.iscecuda.core.stream.event \
        -E test.cxx.iscecuda.core.stream.stream \
        || true \
 && cp Testing/\$(head -1 Testing/TAG)/DynamicAnalysis.xml ."
