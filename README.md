# Underactuated Finger Kineto-Static Analysis

This repository provides the Python implementation accompanying the paper "Numerical Kineto-Static Analysis of an Underactuated Linkage-Driven Robotic Finger". The codebase solves the mechanism's kinematics and static equilibrium directly from its geometric loop-closure equations, transmission matrices, and torsional spring stiffnesses. You can use these scripts to sweep the finger's configuration space, evaluate Jacobian conditioning, compute contact-force distributions under varying actuator torques, and map out feasible grasp regions. For verification, the repository also includes MuJoCo simulation to cross-validate results against dynamic contact.


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

### Usage:
- To generate data, run:
```
python generate_data.py
```

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
