# gen embedding and weight
g++ -O3 -o tmp/gen_data utils/gen_data.cpp
./tmp/gen_data $1 $2 $3 $4 $5

# gen graph data
graph_path="data/graph/graph.txt"
./utils/PaRMAT -nVertices $1 -nEdges $2 -output ${graph_path} > /dev/null
graph_info="$1 $2"
sed -i "1i$graph_info" ${graph_path}
