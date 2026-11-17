# Underactuated Finger Kineto-Static Analysis

This repository provides the Python implementation accompanying the paper "Numerical Kineto-Static Analysis of an Underactuated Linkage-Driven Robotic Finger". The codebase solves the mechanism's kinematics and static equilibrium directly from its geometric loop-closure equations, transmission matrices, and torsional spring stiffnesses. You can use these scripts to sweep the finger's configuration space, evaluate Jacobian conditioning, compute contact-force distributions under varying actuator torques, and map out feasible grasp regions. For verification, the repository also includes MuJoCo simulation to cross-validate results against dynamic contact.


## Repository Layout

```
generate_data.py   Kineto-static analysis: sweeps q1/q2/q3 and the contact-point
                   ratios, writes output_data/data_<timestamp>.csv
models/            MuJoCo model (finger.xml) and phalanx meshes (assets/*.stl)
simulation/        Scripts that run the MuJoCo model
  main.py            pose the finger, apply motor torque, print contact forces, open viewer
  sweep.py           simulate the poses in output_data/output.csv  -> results/sweep/
  finger_torque.py   torque-ramp videos                            -> results/videos/
plotting/          Plots from the analytical / simulated data
  3dplots.py         force surfaces vs. q2/q3                      -> results/figs/
  family.py          min-force family plot                         -> results/family/
  family_gif.py      animated min-force family                     -> results/family/
  errors.py          analytical vs. simulation error               -> results/force_diff/
output_data/       (git-ignored) analytical results, e.g. output.csv
results/           (git-ignored) generated figures, videos, CSVs
tex/               (git-ignored) LaTeX write-up
```

All script paths are resolved relative to the repository root, so the scripts can be run from any directory.

## Getting Started

### Requirements:
- **[Anaconda](https://www.anaconda.com/) *(Optional)*:** Recommended if you prefer managing the application within an isolated virtual environment.

### Project setup:
- Clone the repository.
- Navigate to the project root: 
```
cd underactuated-finger-kinetostatic-analysis
```
- Create and activate conda virtual environment (*Optional*): 
```
conda env create -f environment.yml
conda activate underactuated-finger-kinetostatic-analysis
```
- Install required dependencies:
```
pip install -r requirements.txt
```

MuJoCo 3.1.4 or newer is required (the simulation scripts use `mj_geomDistance`).

### Usage:
- To generate data, run:
```
python generate_data.py
```
- To run the MuJoCo simulation:
```
python simulation/main.py            # interactive viewer with contact forces
python simulation/sweep.py           # needs output_data/output.csv
python simulation/finger_torque.py   # torque-ramp videos
```
- To plot the results:
```
python plotting/3dplots.py           # needs output_data/output.csv
python plotting/family.py            # needs output_data/output.csv
python plotting/family_gif.py        # needs output_data/output.csv
python plotting/errors.py            # needs output_data/output.csv and results/sweep/sim_results.csv
```

The simulation sweep and the plotting scripts read `output_data/output.csv`. That file must carry an `M_mot` column (motor torque) in addition to the columns written by `generate_data.py`, because the scripts group and filter the results by motor torque. Data files are not tracked in git and have to be placed in `output_data/` manually.

## Citation

For the detailed analytical derivation of the underlying kinetostatic equations and kinematic model utilized in this framework, please also refer to our foundational study:

```bibtex
@article{10.1115/1.4072408,
    title   = {Force-Sensitive Underactuated Three-Finger Robot Hand},
    author  = {Savić, Srđan and Čilag, Andrej and Apostolović, Srđan and Nikolić, Milutin and Raković, Mirko},
    journal = {Journal of Mechanisms and Robotics},
    volume  = {18},
    number  = {10},
    pages   = {101001},
    year    = {2026},
    doi     = {10.1115/1.4072408}
}
```

If you use this codebase or build upon our work, please cite our primary numerical study:

```bibtex
The BibTeX entry will be added here once the official proceedings are published.
```
