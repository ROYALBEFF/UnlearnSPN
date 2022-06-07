# UnlearnSPN
## Installation
1. Create a new conda environment from the `conda.yml` file:
```
conda env create -n unlearnspn --file conda.yml
```

2. Activate the conda environment:
```
conda activate unlearnspn
```

3. Unpack the modified version of SPFlow:
```
unzip SPFlow.zip
```

4. Navigate to SPFlow source directory and install it manually:
```
cd SPFlow/src/
python setup.py install
```

5. Unzip datasets:
```
unzip data.zip
```

## Reproducing the results
In order to reproduce the results from the paper, simply run:
```
./run_training_comparison.sh && ./run_experiments.sh
```

The results will be stored using `mlflow`. In order to view the results run
```
mlflow ui
```
in the directory containing the `mlruns` directory. This will start a local web server, which can be accessed through
[127.0.0.1:5000](127.0.0.1:5000) in your web browser.