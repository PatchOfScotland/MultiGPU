make run_cannon
mkdir -p scaling_test
rm -r scaling_test/*

REPEATS=${1:-10}
QUADRANTS=${2:-2}

# These values are approx to increments of 2GB in data
for i in 18258 25820 31622 36514 40824 44722 48304 51640 54772 57736
do
    ./build/default/run_cannon $i ${REPEATS} ${QUADRANTS} > scaling_test/cannon_$i.txt
    echo "Completed run $i"
done

python3 scaling_graph.py scaling_test
