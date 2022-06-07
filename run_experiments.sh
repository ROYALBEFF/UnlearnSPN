declare -a seeds=("250120" "250121" "250122" "250123" "250124" "250125" "250126" "250127" "250128" "250129")
declare -a datasets=("msnbc" "plants" "abalone" "adult" "wine")

for s in "${seeds[@]}"
do
  for d in "${datasets[@]}"
  do
    mlflow run . -e main -P seed="$s" -P dataset="$d" -P remove=100 -P epsilon=0.01 --no-conda;
  done
done