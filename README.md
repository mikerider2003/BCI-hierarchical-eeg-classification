# BCI-hierarchical-eeg-classification

## Data

This project uses the [`ds007537`](https://openneuro.org/datasets/ds007537/versions/1.0.0) dataset as a DataLad/git-annex dataset.  
After cloning the repository, the visible files may only be lightweight placeholders. To download the actual data files, use `datalad get`.

### Setup Instructions

```bash
# Create the data directory
mkdir data

# Clone the dataset into the data directory
cd data
git clone https://github.com/OpenNeuroDatasets/ds007537.git

# Navigate to the dataset and fetch data using datalad
cd ds007537
```

### Download the full dataset

```bash
datalad get .
```

### Download a specific subject

```bash
datalad get sub-01
```
