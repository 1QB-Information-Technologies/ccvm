# Pull Request

Following features were added to the CCVM simulator

1. “lr” replaced with “dt” in all scripts in the directory: `examples/` and the solvers in `ccvm_simulators/solvers`'re updated correspondingly.

2. A class `LangevinSolver` was added in the module `ccvm_simulators/solvers/langevin_solver.py`

3. Adam algorithm was implemented as a special method `__call(...)__` in the classes `DLSolver`, `MFsolver`, and `LangevinSolver` in `ccvm_simulators/solvers/`  

4. Demo scripts were created in `examples/` associated with each aforementioned solver 

5. `readme.md` in `examples/` was updated


# Issues 
2. `readme.md` in `ccvm/examples/` was updated
    1. Instance requires update after fixing issue
