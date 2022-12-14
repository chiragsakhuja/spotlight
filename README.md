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

Create an Anaconda environment and activate it.  Then build MAESTRO and its
wrapper.
```
conda env create -f environment.yml
conda activate spotlight-ae
scons -j`nproc`
```

## Docker Setup
Build the Docker image (takes about 20 minutes).
```
docker build -t spotlight .
```

Open an interactive shell in a new container.
```
docker run -it spotlight /bin/bash
```

Within the Docker container, activate the Anaconda environment and build MAESTRO
and its wrapper.  Spotlight should be run (see the following section) within the
Docker container as well.
```
conda activate spotlight-ae
scons -j`nproc`
```

To resume work in the container after stopping/exiting, run the following
commands.
```
docker container ls -a                    (to get Container ID)
docker start <Container ID>
docker exec -it <Container ID> /bin/bash
```

# Running Spotlight
We provide a script, `run-ae.sh`, that runs the experiments in the paper.  We
also provide a script, `compare-ae.sh`, that aggregates the results into a CSV
file for easy comparison.

## `run-ae.sh`
There are a few modes that the script can run in: Single, Main-Edge, Main-Cloud,
General, and Ablation.

### Single Mode (Runs in minutes)
This mode runs a single configuration of Spotlight, where a configuration is
defined as a specific scale, search algorithm, DL model, and optimization target
(EDP or Delay).  In either case, results are written to the `results` directory.

For example, to optimize ResNet-50 for EDP on an edge-scale accelerator using
Spotlight, run the following command.
```
./run-ae.sh single --model RESNET --target EDP --technique Spotlight --scale Edge [--trials 1]
```

The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.

### Main Modes (Runs in multiple days)
These modes, Main-Edge and Main-Cloud, are intended to reproduce Figures 6 and 7
in the paper.  They run experiments with Spotlight and the hand-designed
accelerators.  Because there are so many configurations---4 techniques, 5 DL
models, and 2 optimization targets, equaling a total of 40 configurations---it
can take nearly a day to complete a full run, even if only 1 trial of each
configuration is used.  Moreover, it is more difficult to explore the large
cloud-scale space, so a full run at cloud-scale can take multiple days to
complete.  This mode is parallelized, so it is beneficial to have at least 8
cores available.

To run both main modes, run the following commands.
```
./run-ae.sh main-edge [--trials 1]
./run-ae.sh main-cloud [--trials 1]
```

The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.  It is recommended to keep the trials at 1
for this mode, though we use 10 for the results in the paper.

### General Mode (Runs in minutes)
This mode is similar to the main modes except that it is designed to reproduce
the data for Figure 9, which demonstrates that Spotlight's accelerators are
fine-tuned but still generalize well.  Specifically, this mode runs two
experiments: (1) It uses an accelerator that was fine-tuned to run all 5 DL
workloads, and (2) It uses an accelerator that was fine-tuned to run VGG16,
ResNet-50, and MobileNetV2, and it shows that performance is still good when the
accelerator runs MNasNet and Transformer.

To run general mode, run the following command.
```
./run-ae.sh general [--trials 1]
```

The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.

### Ablation Mode (Runs in multiple hours)
This mode is similar to the main modes except it is designed to reproduce the
data for Figure 10, which compares the performance of different variants of
Spotlight.  Specifically, this mode runs Spotlight-GA, Spotlight-R, Spotlight-V,
and Spotlight-F, and it assumes that Spotlight has already been run through a 
different mode.  This mode will take several hours to complete, and it is also
parallelized, so it is beneficial to have at least 8 cores available.

To run ablation mode, run the following command.
```
./run-ae.sh ablation [--trials 1]
```

The `--trials` argument is optional and dictates how many independent trials to
run Spotlight for.  The default is 1.

## `compare-ae.sh`
This script aggregates the results in the `results` directory for easy
comparison with the figures in the paper.  Specifically, for each configuration,
this script collects the Minimum, Maximum, and Median performance metrics for
each configuration.  Furthermore, the script normalizes each configuration to
Spotlight.  The type of comparison is specified directly after the script.

The script runs as follows, where only one comparison type is selected.
```
./compare-ae.sh main-edge|main-cloud|general|ablation
```

## Running Spotlight Directly
Spotlight can be run directly through Python.  To see the full suite of command
line options that Spotlight provides, run the following command from
SPOTLIGHT_ROOT.
```
python src/main.py --help
```