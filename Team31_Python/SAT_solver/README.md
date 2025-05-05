DPLL_SAT_Solver - README

## Directory Structure
<!-- Team31_Python/
└── SAT_Solver/
    ├── SATLIB_Benchmarks/              # Directory containing CNF benchmark files
    │   └── *.cnf                       
    ├── Baseline_SAT_Solver.py         # Main Python file 
    ├── uf20-01.cnf                    # Sample input file in DIMACS CNF format
    ├── README.md                       
    └── Team31_ECE51216_DPLL_SAT_SOLVER.pdf  # Project report -->
- Team31_Python/SAT_Solver/SATLIB_Benchmarks/*.cnf (benchmark files)
- Team31_Python/SAT_Solver/Baseline_SAT_Solver.py 
- Team31_Python/SAT_Solver/uf20-01.cnf  
- Team31_Python/SAT_Solver/README.md
- Team31_Python/SAT_Solver/Team31_ECE51216_DPLL_SAT_SOLVER.pdf

## CODE compilation and execution

- Benchmarks:
  Each file is SATLIB Benchmark problems of varying size of literals ranging from 20 to 200 and clauses from 91 to 860 corresponding to number of literals with both SAT and UNSAT. Benchmark uf20-01.cnf is copied to same folder as python file to execute directly.
  
- Execution:
```bash
 python3 Baseline_SAT_Solver.py uf20-01.cnf
```
## Functions and Data structures used

- Functions:
  
- Python Data Structures:
- **List** : Used for storing clauses, literals, watched lists, learned clauses, and propagated literals due to their ordered and mutable nature.
- **Set** : Ensures uniqueness of variables in the formula with fast membership checks.
- **Deque** : Efficient for stack and queue operations; used for the assignment stack and unit clause queue.
- **Tuple** : Employed to return multiple values from functions like `cdcl()`, offering immutability and lightweight structure.
- **Dictionary** : Maps literals to watched clause lists for quick lookup and clause management.
- **NumPy Arrays** : Used for performance-efficient tracking of literal activity via initialized zero arrays for VSIDS scoring.
