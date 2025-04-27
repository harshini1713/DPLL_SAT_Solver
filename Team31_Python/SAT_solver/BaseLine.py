import os
from typing import List, Dict

def parse_dimacs(file_path: str) -> List[List[int]]:
    clauses = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line[0] in ('c', 'p', '%', '0'):  # Skip comments and problem definitions
                continue
            try:
                clause = list(map(int, line.split()))
                if clause[-1] == 0:
                    clause.pop()  # Remove trailing zero
                clauses.append(clause)
            except ValueError:
                print(f"Skipping malformed line in {file_path}: {line}")
    return clauses

def is_satisfied(clause: List[int], assignment: Dict[int, int]) -> bool:
    return any(lit in assignment and assignment[lit] for lit in clause)

def all_satisfied(clauses: List[List[int]], assignment: Dict[int, int]) -> bool:
    return all(is_satisfied(clause, assignment) for clause in clauses)

def select_unassigned(clauses: List[List[int]], assignment: Dict[int, int]) -> int:
    for clause in clauses:
        for lit in clause:
            if abs(lit) not in assignment:
                return abs(lit)
    return 0

def dpll(clauses: List[List[int]], assignment: Dict[int, int]) -> bool:
    if all_satisfied(clauses, assignment):
        return True
    
    var = select_unassigned(clauses, assignment)
    if var == 0:
        return False
    
    for value in [1, 0]:
        assignment[var] = value
        if dpll(clauses, assignment):
            return True
        del assignment[var]
    
    return False

def process_benchmarks(folder_path: str):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".cnf"):  # Process only CNF files
            file_path = os.path.join(folder_path, filename)
            cnf_formula = parse_dimacs(file_path)
            assignment = {}
            
            result = "SAT" if dpll(cnf_formula, assignment) else "UNSAT"
            results.append(f"{filename}: {result}")
            
            print(f"{filename}: {result}")
            if result == "SAT":
                print("ASSIGNMENT:", ", ".join(f"{k} = {v}" for k, v in assignment.items()))
    
    print("\nFinal Results:")
    for res in results:
        print(res)

if __name__ == "__main__":
    folder_path = "D:\ms sem 2 docs\dsda\DPLL SAT SOLVER\Team31_Python\SAT_solver\Benchmarks"  # Change this to your actual folder path
    process_benchmarks(folder_path)
