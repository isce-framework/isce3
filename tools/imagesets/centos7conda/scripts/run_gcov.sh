# Check if one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Error: missing 1 argument, the directory path of the test data"
    exit
fi
datadir=$1

# Check if data directory exists
if [ ! -d "$datadir" ] 
then
    echo "Error: directory $datadir does not exist." 
    exit 
fi

set -x
docker run \
          --rm \
          --mount type=bind,source=$datadir,target=/tmp/data \
          -w /tmp/data \
          -u $UID:$(id -g) \
          -i --tty nisar-adt/isce3:centos7conda \
          sh -ci "mkdir -p output_gcov scratch_gcov
                  time python3 -m pybind_nisar.workflows.gcov run_config_gcov_v2.yaml"
