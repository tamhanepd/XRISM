# XRISM

This repository contains the code and data file for running the script to fit gas emission lines in a spectrum. The code is written in Python and the data file is provided as a csv file with four columns for x and y axis values and their errors.

## Prerequisites
To run the script, you need to have the following software and packages installed:

- Python 3+
- numpy
- matplotlib
- lmfit

## Getting Started
To get started, follow the steps below:

1. Clone this repository to your local machine.
2. Install the necessary dependencies mentioned in the Prerequisites section.
3. Place the data file [image_nh0p041_v100_exp1000_Z0p5.csv] in the designated directory.
4. Run the script [Fit_lines_new.py].

## Usage

```shell
$ python Fit_lines_new.py
```

## Data File
The data file [image_nh0p041_v100_exp1000_Z0p5.csv] contains count rate and energy values and their errors generated for Abell 1795 for 1000 ks long exposure simulation assuming 0.5 solar metallicity, 100 km/s turbulent velocity in the gas and nH of 0.041. The first column is the energy in keV (x), second column is the error in energies in keV (dx), third column is the count rate per keV (y) and last column is its error (dy). Make sure to place it in the same directory as the script before running the script. Otherwise, change the path of the data file in the script.
