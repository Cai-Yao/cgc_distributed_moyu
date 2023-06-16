# for debug
# set -x 

# check arg
if [ $# -ne 5 ]; then
  echo "arg numbber is wrong, need five: |V|, |E|, F0, F1, F2"
  exit 1
fi

# mkdir
if [ ! -d "data/embedding" ];then
    mkdir -p data/embedding
fi
if [ ! -d "data/weight" ];then
    mkdir -p data/weight
fi
if [ ! -d "data/graph" ];then
    mkdir -p data/graph
fi
if [ ! -d "tmp" ];then
    mkdir -p tmp
fi

# gen data
echo "begine gen graph data, embedding, weight"
sh run_gen_data.sh $1 $2 $3 $4 $5
echo "finish gen graph data, embedding, weight"

# build
echo "begin build"
cd distributed_moyu
make > /dev/null
cd ..
mv distributed_moyu.exe tmp/
echo "finish build"

# execute
echo "begin execute"
./tmp/distributed_moyu.exe $3 $4 $5 "data/graph/graph.txt" "data/embedding/${1}.bin" "data/weight/W1_${3}_${4}.bin" "data/weight/W2_${4}_${5}.bin" > tmp/distributed_moyu.txt
./utils/standard $3 $4 $5 "data/graph/graph.txt" "data/embedding/${1}.bin" "data/weight/W1_${3}_${4}.bin" "data/weight/W2_${4}_${5}.bin" > tmp/standard.txt
echo "finish execute"

# check result
result1=$(head -n 1 tmp/distributed_moyu.txt)
result2=$(head -n 1 tmp/standard.txt)

if [ "$result1" != "$result2" ]; then
    echo "output result is wrong"
else
    echo "output result is right"
    echo "standard run time result: $(tail -n 1 tmp/standard.txt)"
    echo "new rum time result: $(tail -n 1 tmp/distributed_moyu.txt)"
fi

# clean
rm -rf data/ tmp/