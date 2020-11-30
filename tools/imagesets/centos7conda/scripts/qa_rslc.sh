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
          sh -ci "mkdir -p qa_rslc
                  time verify_rslc.py --fpdf qa_rslc/graphs.pdf \
                      --fhdf qa_rslc/stats.h5 --flog qa_rslc/qa.log --validate \
                      --quality output_rslc/rslc.h5
                  time cfchecks.py output_rslc/rslc.h5"
