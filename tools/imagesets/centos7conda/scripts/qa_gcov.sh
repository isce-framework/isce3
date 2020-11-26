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
          -i --tty nisar-adt/isce3:centos7conda-nisar \
          sh -ci "mkdir -p qa_gcov
                  time python3 /opt/QualityAssurance/verify_gcov.py --fpdf qa_gcov/graphs.pdf \
                      --fhdf qa_gcov/stats.h5 --flog qa_gcov/qa.log --validate \
                      --quality output_gcov/gcov.h5
                  time python3 /opt/CFChecker/src/cfchecker/cfchecks.py output_gcov/gcov.h5"
