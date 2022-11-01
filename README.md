# Spotlight
Spotlight is an automated hardware-software co-design tool for deep learning
accelerators.  The inputs are (1) one or more deep learning models and (2) a
hardware budget, and the outputs are (1) the architectural parameters of a
fine-tuned deep learning accelerator and (2) an optimized software schedule for
each layer of the input model(s).

## Directory Structure
Spotlight is a python framework that builds on top of the analytical model,
[MAESTRO](https://maestro.ece.gatech.edu/).  The directory structure is as
follows:
```
/                     : SPOTLIGHT_ROOT
|-- maestro-wrapper
|   |-- maestro       : MAESTRO with minor modifications to support Spotlight
|   |-- *.cpp/*.hpp   : Wrapper files around MAESTRO
|-- src               : Python source for Spotlight
```

# Requirements
Spotlight can either be built natively or within a provided Docker container.

## Native Setup
Spotlight requires the following packages to be installed natively:
1. A C++ compiler that supports the C++17 standard
2. Boost libraries
3. Python 3.9 or later
4. Anaconda
5. SCons build system

Create an Anaconda environment and activate it.
```
conda env create -f environment.yml
conda activate spotlight-ae
```

Build MAESTRO and the MAESTRO wrapper.
```
scons -j`nproc`
```

## Docker Setup
TODO

# Running Spotlight
We provide a script, `run-ae.sh`, that runs the most common configurations, as
presented in our paper, of Spotlight.  We also provide a script,
`compare-ae.sh`, that aggregates the results into a CSV file of all runs.

## `run-ae.sh`
There are two modes that the script can run in: Single and Full.

### Single Mode
This mode runs a single configuration of Spotlight, where a configuration is
defined as a specific search algorithm, DL model, and optimization target (EDP
or Delay).  In either case, results are written to the `results` directory.

For example, to optimize ResNet-50 for EDP using Spotlight, run the following
command.
```
./run-ae.sh single --model RESNET --target EDP --algorithm Spotlight [--trials 1]
```

The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.

### Full Mode
To run the full suite of search algorithms, DL models, and optimization targets,
run with this mode.  Note that, even when evaluating just 1 trial of each
configuration, this mode can take a **very long time**.  There are 5 search
algorithms, 5 DL models, and 2 optimization targets, equaling a total of 50
configurations, each of which can take multiple hours to complete.  This mode
does parallelize the runs, but it can still take one or more days to collect all
results.

To run in full mode, use the following command.
```
./run-ae.sh full [--trials 1]
```
The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.

## Running Spotlight Directly
Spotlight can be run directly through Python.  To see the full suite of command
line options that Spotlight provides, run the following command from
SPOTLIGHT_ROOT.
```
python src/main.py --help
```

## `compare-ae.sh`
This script aggregates all the results in the `results` directory.
Specifically, for each configuration, this script collects the Minimum, Maximum,
and Median performance metrics for each configuration.  Furthermore, the script
normalizes each configuration to Spotlight.

The script does not take any arguments and can simply be run as follows.
```
./compare-ae.sh
```