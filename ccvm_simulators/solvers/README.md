# Solvers

### Solver classes
There are currently two methods available for each solver class [DL-CCVM, MF-CCVM, Langevin] which are **original** (default) and **Adam**.

- `dl_solver.py`: models the delay line coherent continuous-variable machine (DL-CCVM).
- `mf_solver.py`: models the measurement-feedback continuous-variable machine (MF-CCVM).
- `langevin_solver.py`: models typical Langevin dynamics as a system of SDE (Langevin).


#### The solver hierarchy 
The diagram portrays the relationships among different solvers and their
association with the abstracted CCVM solver class.

<p align="center">
    <img src="https://github.com/1QB-Information-Technologies/ccvm/blob/main/diagrams/solver_%20hierarchy.png?raw=true">
</p>
