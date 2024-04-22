# HDN-DDI: drug-drug interactions prediction with hierarchical molecular graphs and enhanced dual-view representation learning

**Authors**: Jinchen Sun, Haoran Zheng*

**Article Link**: None (the url will be given after the paper is accepted.)

# Requirement
To run the code, you need the following dependencies:
* PyTorch >= 1.9.0
* PyTorch Geometry == 2.0.3
* rdkit == 2020.09.2

# Reimplement
The average performace of HDN-DDI can be calculated by `evaluate.ipynb` and the training code will be supplemented after the paper is accepted.

# Dataset & basic data required to run HDN-DDI
Please ensure the working path is `.*/HDN-DDI/` and download the zip file from [Google Drive](https://drive.google.com/file/d/15IN_WsI92UwytHM1urCMAJC2WDBjKgDY/view?usp=sharing)

Then, run the code in terminal:
```
unzip Data-for-HDN-DDI.zip
mv drugbank/ drugbank_test/
mv inductive_data/ drugbank_test/
mv twosides/ twosides_test/
rm Data-for-HDN-DDI.zip
```
