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
  **partial_assignment_literals**: This function returns the unassigned literals from a clause if none of its literals are already satisfied by the current assignment. If any literal in the clause is satisfied, it returns an empty list immediately.

  **watched_literal_update_pass**: This function updates watched literals in a clause when a new variable is assigned, checking whether the clause is satisfied, unsatisfied, or becomes a unit clause. It returns a tuple indicating the clause status, the key literal involved, and whether unit propagation is needed.

  **is_satisfied**: This function checks if the clause is satisfied under the current assignment using the two watched literals. It returns `True` if either watched literal is assigned in a way that makes the clause true.

  **literal_assignment**: This function assigns a literal at a given decision level and updates all affected watched literals accordingly. It detects conflicts or new unit clauses, queues them for propagation, and returns whether the assignment was successful.

  **all_assigned**: This function checks whether all variables in the problem have been assigned a value. It returns `True` if the number of assigned variables matches the total number of variables.

  **backtrack**: This function undoes variable assignments made after a specified decision level by popping from the assignment stack. It resets the assignment, decision level, and previous value for each backtracked variable.

  **conflict_inspection**: This function performs conflict analysis using clause learning to identify an assertive clause when a conflict occurs during SAT solving. It resolves conflicting clauses, updates activity scores, computes the assertion level and LBD (Literal Block Distance), and returns the new backtrack level for non-chronological backtracking.

  **unit_propagation**: The `unit_propagation` function propagates unit clauses by assigning their literals and checking for conflicts, returning either all propagated literals or a conflicting clause. 

  **vsids_heuristic**: The `vsids_heuristic` function selects the next decision literal based on VSIDS, prioritizing the unassigned variable with the highest activity score.

  **most_recurring_heuristic**: This function selects the unassigned literal that appears most frequently in unsatisfied clauses using a simple frequency-based heuristic. It returns the literal (positive or negative) that maximizes clause occurrences for potential decision-making.

  **random_heuristic**: This function randomly selects an unassigned variable, assigns it a random polarity (positive or negative), and removes it from the internal unassigned list. It ensures randomness in decision-making for SAT solvers when no specific heuristic is applied.

  **select_decision_literal**: This function selects a decision literal based on the specified heuristic (VSIDS, most recurring, or random). It raises an error if an unknown heuristic value is provided.

  **cdcl**: The `cdcl` function implements a conflict-driven clause learning (CDCL) algorithm to solve a CNF formula by making decisions, propagating units, and handling conflicts with backtracking and learned clauses. It tracks decisions, unit propagations, and reinstates while managing conflict limits and literal block distance for clause deletion.

  **delete_learned_clauses_by_lbd**: This function removes learned clauses with a literal block distance (LBD) greater than a specified limit from the watched lists. It retains clauses with lower LBD in the learned clauses list and updates the watched lists accordingly.

  **find_model**: The `find_model` function reads a CNF formula from a DIMACS file, applies the CDCL algorithm to solve it, and returns the result, including whether the formula is satisfiable, the model (variable assignments), CPU time, and statistics on decisions, unit propagations, and reinstates. It handles file reading errors and provides a formatted output.

- Python Data Structures:
**List** : Used for storing clauses, literals, watched lists, learned clauses, and propagated literals due to their ordered and mutable nature.
**Set** : Ensures uniqueness of variables in the formula with fast membership checks.
**Deque** : Efficient for stack and queue operations; used for the assignment stack and unit clause queue.
**Tuple** : Employed to return multiple values from functions like `cdcl()`, offering immutability and lightweight structure.
**Dictionary** : Maps literals to watched clause lists for quick lookup and clause management.
**NumPy Arrays** : Used for performance-efficient tracking of literal activity via initialized zero arrays for VSIDS scoring.
