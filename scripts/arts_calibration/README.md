# ARTS for HAMP

This README gives a quick introduction on how to use ARTS for calculating brightness temperatures at the HAMP radiometer frequencies from dropsonde data.

### Getting Started:
1. Create a conda environment with
```conda env create -f environment.yaml```
2. Check whether you can run ```arts_bt_calculation.py```
3. You can run the analysis script ```arts_bt_calculation.py``` and loop over all flights within the script, or you can submit it as a batch job for each flight using ```master_submitter.py```. Then you will need to delete the loop over flights in ```arts_bt_calculation.py```.

### Code Structure
```arts_bt_calculation.py``` is the main script to run the brightness temperature calculations. It reads in dropsonde data, sets up the ARTS simulation, and outputs brightness temperatures. You need to run this script from the root directory of the repository.

```analyse_bt_diffs.py``` is a script to analyse the differences between measured and simulated brightness temperatures. It reads in the output from ```arts_bt_calculation.py``` and compares it with HAMP measurements.
