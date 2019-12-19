# eICU Benhmarks
## Reference
Seyedmostafa Sheikhalishahi and Vevake Balaraman and Venet Osmani.
"Benchmarking machine learning models on eICU critical care dataset" available on arXiv (https://arxiv.org/abs/1910.00964v1)

## Citation
First of all be sure to cite [eICU paper!](https://www.nature.com/articles/sdata2018178)

If you use this code or these benchmarks in your research, please cite the following publication.
> @misc{sheikhalishahi2019benchmarking,
    title={Benchmarking machine learning models on eICU critical care dataset},
    author={Seyedmostafa Sheikhalishahi and Vevake Balaraman and Venet Osmani},
    year={2019},
    eprint={1910.00964},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}




## Requirements
You must have the csv files of eICU on your local machine
### Packages
* numpy==1.15.0
* scipy==1.2.0
* scikit-learn==0.21.2
* pandas==0.24.1

For Feedforward Network and LSTM:
* Keras==2.2.4

## Structure
The content of this repository can be divide into two parts:

* data extraction
* running the models (baselines, LSTM)

## How to Build this benchmark
Here are the required steps to create the benchmark. The eICU dataset CSVs should be available on the disk.

1. Clone the repository.
> git clone https://

> cd 

2. The following command generates one directory per each patient and writes patients demographics into pats.csv, the items extracted from Nursecharting into nc.csv and the lab items into lab.csv


3. The following command takes into account the three produced csv files for each patient and create a csv for each patient with all the required clinical variables in a time series method (each time step is one hour).


4. The following command trains the baseline or the LSTM network with the desired configuration.(E.g The use of categorical variables, the usage of numerical variables, the usage of one-hot encoding or embedding, which baseline to choose ann or lr).

## Data extraction
The data extraction is divided into two steps:

1. Extract the all the available clinical variables in order to apply filter for each task.

2. Extract data for each single task using the extracted data in the previous step, which will be done automatically in the train.py file.

### How to extract:
* Run python data_extraction_root.py with the args and save the data into output directory in config file (all_data.csv will be saved as the name of dataset)

## How to run each experiment
The experiments are divided into two scripts. In the both scripts there are arguments related to task, numerical, categorical, artificial neural networks, one-hot encoding, and mortality window data. Those arguments can be provided as binary and for mortality window we consider the first 24 and 48 hours of the admission data.

The baseline experiments can be ran by running python baseline.py --args

The LSTM experiments can be ran by running python train.py --args

## Motivation


