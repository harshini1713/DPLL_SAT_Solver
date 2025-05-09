from typing import Tuple, Optional, List
from collections import deque
import numpy as np
import random
import argparse
import time

class Clause_CNF:
    
    def __init__(self, literals, watched_one=None, watched_two=None, learned=False, literal_block_distance=0):
        self.literals = literals  
        self.size = len(self.literals)
        self.watched_one = watched_one
        self.watched_two = watched_two
        self.learned = learned
        self.literal_block_distance = literal_block_distance

        if (not watched_one) and (not watched_two):
            if len(self.literals) > 1:
                self.watched_one = 0
                self.watched_two = 1

            elif len(self.literals) > 0:
                self.watched_one = self.watched_two = 0
    @property
    def literal_block_distance(self):
        return self._literal_block_distance

    @literal_block_distance.setter
    def literal_block_distance(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("LBD must be non-negative")
        self._literal_block_distance = value

    
    def partial_assignment_literals(self, assignment: list) -> list:
    # Return an empty list immediately if any literal is already assigned (i.e., satisfied).
       if any(assignment[abs(literal)] == literal for literal in self.literals):
        return []

    # filter to find unassigned literals
       unassigned = filter(lambda literal: assignment[abs(literal)] == 0, self.literals)
       return list(unassigned)


    def watched_literal_update_pass(self, assignment: list, new_variable: int) -> Tuple[bool, int, Optional[int]]:
       
        #  prev watched literal index: `self.watched_one`
        if new_variable == abs(self.literals[self.watched_two]):
            temp = self.watched_one
            self.watched_one = self.watched_two
            self.watched_two = temp

        # If Clause[self.watched_one] is True in this new variable assignment or
        # Clause[self.watched_two] has been True previously, then the Clause is satisfied
        if (self.literals[self.watched_one] == assignment[abs(self.literals[self.watched_one])] or
                self.literals[self.watched_two] == assignment[abs(self.literals[self.watched_two])]):
            return True, self.literals[self.watched_one], False

        # If Clause[self.watched_one] is False in this new variable assignment and
        # Clause[self.watched_two] is also False from previous assignment, then the Clause is unsatisfied
        if (-self.literals[self.watched_one] == assignment[abs(self.literals[self.watched_one])] and
                -self.literals[self.watched_two] == assignment[abs(self.literals[self.watched_two])]):
            return False, self.literals[self.watched_one], False

        # If Clause[self.watched_one] is False in this new variable assignment and
        # Clause[self.watched_two] is still unassigned, then look for new index of the watched literal `self.watched_one`
        if (-self.literals[self.watched_one] == assignment[abs(self.literals[self.watched_one])] and
                assignment[abs(self.literals[self.watched_two])] == 0):
            old_watched_one = self.watched_one
            for w in [(self.watched_one + i) % self.size for i in range(self.size)]:
                # new index `w` must not be equal to `self.watched_two` and
                # Clause[w] cannot be False in the current assignment
                if w == self.watched_two or -self.literals[w] == assignment[abs(self.literals[w])]:
                    continue

                self.watched_one = w
                break

            # If the new watched literal index `self.watched_one` not found ---> Clause is unit with
            # Clause[self.watched_two] being the only unassigned literal.
            if self.watched_one == old_watched_one:
                return True, self.literals[self.watched_one], True

            # Otherwise the state of the Clause is either not-yet-satisfied or satisfied -> both not important
            return True, self.literals[self.watched_one], False

    def is_satisfied(self, assignment: list) -> bool:
        
        return (self.literals[self.watched_one] == assignment[abs(self.literals[self.watched_one])] or
                self.literals[self.watched_two] == assignment[abs(self.literals[self.watched_two])])
        
class CNF_Formula:
    def __init__(self, formula):
        self.formula = formula
        self.clauses = [Clause_CNF(literals) for literals in formula]
        self.learned_clauses = []
        self.unit_clauses_queue = deque()
        self.assignment_stack = deque()
        self.watched_lists = {}
        self.variables = set()

        for clause in self.clauses:
            # Add unit clause to queue
            if clause.watched_one == clause.watched_two:
                self.unit_clauses_queue.append((clause, clause.literals[clause.watched_two]))

            # Filter watched literals
            watched_literals = filter(lambda lit: lit in (clause.literals[clause.watched_one], clause.literals[clause.watched_two]), clause.literals)

            # Process filtered literals
            for literal in watched_literals:
                var = abs(literal)
                self.variables.add(var)
                self.watched_lists.setdefault(var, []).append(clause)

        max_var = max(self.variables)
        self.assignment = [0] * (max_var + 1)
        self.previous = [None] * (max_var + 1)
        self.decision_level = [-1] * (max_var + 1)
        self.positive_literal_counter = np.zeros(max_var + 1, dtype=np.float64)
        self.negative_literal_counter = np.zeros(max_var + 1, dtype=np.float64)

    def literal_assignment(self, literal: int, decision_level: int) -> Tuple[bool, Optional[Clause_CNF]]:
     var = abs(literal)

     self.assignment_stack.append(literal)
     self.assignment[var] = literal
     self.decision_level[var] = decision_level

     watched_list = list(self.watched_lists[var])
     existing_unit_literals = set(map(lambda x: x[1], self.unit_clauses_queue))

     for clause in watched_list:
        success, new_watched_literal, is_unit = clause.watched_literal_update_pass(self.assignment, var)

        if not success:
            return False, clause

        new_var = abs(new_watched_literal)
        if new_var != var:
            # Move clause to new watched list
            if new_var not in self.watched_lists:
              self.watched_lists[new_var] = []
            self.watched_lists[new_var].append(clause)
            self.watched_lists[var] = list(filter(lambda c: c != clause, self.watched_lists[var]))

        if is_unit:
            new_unit_lit = clause.literals[clause.watched_two]
            if new_unit_lit not in existing_unit_literals:
                self.unit_clauses_queue.append((clause, new_unit_lit))
                existing_unit_literals.add(new_unit_lit)

     return True, None

    def all_assigned(self) -> bool:
       
        return len(self.variables) == len(self.assignment_stack)

    def backtrack(self, decision_level: int) -> None:
     
     while self.assignment_stack:
        literal = self.assignment_stack[-1]
        var = abs(literal)
        if self.decision_level[var] <= decision_level:
            break
        self.assignment_stack.pop()
        self.assignment[var] = 0
        self.previous[var] = None
        self.decision_level[var] = -1

    @staticmethod
    def resolve(clause_one: list, clause_two: list, literal: int) -> list: 
        input_clause1 = set(clause_one)
        input_clause2 = set(clause_two)
        if -literal in input_clause1:
           input_clause1.remove(-literal)
        if literal in input_clause2:
           input_clause2.remove(literal)
        return list(input_clause1.union(input_clause2))

    def conflict_inspection(self, previous_of_conflict: Clause_CNF, decision_level: int) -> int:

      # Conflict at decision level 0 ---> return -1
      if decision_level == 0:
        return -1

      assertive_clause_literals = previous_of_conflict.literals
      current_assignment = deque(self.assignment_stack)

      # Conflict resolution using clause learning
      while len(list(filter(lambda l: self.decision_level[abs(l)] == decision_level, assertive_clause_literals))) > 1:
        while True:
            literal = current_assignment.pop()
            if -literal in assertive_clause_literals:
                assertive_clause_literals = self.resolve(
                    assertive_clause_literals,
                    self.previous[abs(literal)].literals,
                    literal
                )
                break

      # Initialize for LBD and watched literal determination
      unit_literal = None
      watched_two = None
      decision_level_present = [False] * (decision_level + 1)
      assertion_level = 0
      # Update activity counters and gather assertion level info
      for index, literal in enumerate(assertive_clause_literals):
        var = abs(literal)
        level = self.decision_level[var]

        if level == decision_level:
            unit_literal = literal
            watched_two = index

        if 0 < level < decision_level:
            assertion_level = max(level, locals().get('assertion_level', 0))

        if not decision_level_present[level]:
            decision_level_present[level] = True

        # Apply exponential decay and bump literal scores
        self.positive_literal_counter *= 0.9
        self.negative_literal_counter *= 0.9
        if literal > 0:
            self.positive_literal_counter[literal] += 1
        else:
            self.negative_literal_counter[var] += 1

      # Compute LBD using set + map + lambda
      literal_block_distance = len(set(map(lambda l: self.decision_level[abs(l)], assertive_clause_literals)))

      # Find watched_one at assertion level
      watched_one = None
      if len(assertive_clause_literals) > 1:
        current_assignment = deque(self.assignment_stack)
        while current_assignment:
            literal = current_assignment.pop()
            if self.decision_level[abs(literal)] == assertion_level:
                match = list(filter(
                    lambda i: abs(assertive_clause_literals[i]) == abs(literal),
                    range(len(assertive_clause_literals))
                ))
                if match:
                    watched_one = match[0]
                    break
      else:
        watched_one = watched_two

      # Create and register the assertive clause
      assertive_clause = Clause_CNF(assertive_clause_literals,watched_one=watched_one,watched_two=watched_two,learned=True,literal_block_distance=literal_block_distance)

      self.watched_lists[abs(assertive_clause.literals[watched_one])].append(assertive_clause)
      if watched_one != watched_two:
        self.watched_lists[abs(assertive_clause.literals[watched_two])].append(assertive_clause)

      self.learned_clauses.append(assertive_clause)

      self.unit_clauses_queue.clear()
      self.unit_clauses_queue.append((assertive_clause, unit_literal))

      return assertion_level

    def unit_propagation(self, decision_level: int) -> Tuple[list, Optional[Clause_CNF]]:
        
        propagated_literals = []
        while self.unit_clauses_queue:
            unit_clause, unit_clause_literal = self.unit_clauses_queue.popleft()
            propagated_literals.append(unit_clause_literal)
            self.previous[abs(unit_clause_literal)] = unit_clause

            success, previous_of_conflict = self.literal_assignment(unit_clause_literal, decision_level)
            if not success:
                return propagated_literals, previous_of_conflict

        return propagated_literals, None

    # def vsids_heuristic(self) -> int:
    #   decision_literal = None
    #   best_counter = -1  # use -1 to ensure 0 counts are considered

    #   for variable in self.variables:
    #     if self.assignment[variable] == 0:
    #         pos_score = self.positive_literal_counter[variable]
    #         neg_score = self.negative_literal_counter[variable]

    #         if pos_score >= neg_score and pos_score > best_counter:
    #             decision_literal = variable
    #             best_counter = pos_score
    #         elif neg_score > pos_score and neg_score > best_counter:
    #             decision_literal = -variable
    #             best_counter = neg_score

    #   return decision_literal
    def vsids_heuristic(self) -> int:
        
         decision_literal = None
         best_counter = -1
         for variable in self.variables:
            if self.assignment[variable] == 0:
                if self.positive_literal_counter[variable] > best_counter:
                    decision_literal = variable
                    best_counter = self.positive_literal_counter[variable]

                if self.negative_literal_counter[variable] >= best_counter:
                    decision_literal = -variable
                    best_counter = self.negative_literal_counter[variable]

         return decision_literal

    def most_recurring_heuristic(self) -> int:
        
        number_of_clauses = -1
        decision_literal = None
        for variable in self.variables:
            if self.assignment[variable] == 0:
                positive_clauses = 0
                negative_clauses = 0
                for clause in self.watched_lists[variable]:
                    if not clause.is_satisfied(self.assignment):
                        unassigned = clause.partial_assignment_literals(self.assignment)
                        if variable in unassigned:
                            positive_clauses += 1

                        if -variable in unassigned:
                            negative_clauses += 1
                if positive_clauses > number_of_clauses and positive_clauses > negative_clauses:
                    number_of_clauses = positive_clauses
                    decision_literal = variable

                if negative_clauses > number_of_clauses:
                    number_of_clauses = negative_clauses
                    decision_literal = -variable

        return decision_literal
        
    def random_heuristic(self) -> int:
       
       # If _unassigned is not initialized or empty, populate it using filter
       if not hasattr(self, '_unassigned') or not self._unassigned:
        self._unassigned = list(filter(lambda var: self.assignment[var] == 0, self.variables))

       # If no unassigned variables are found, raise an exception
       if not self._unassigned:
        return None
        #raise Exception("No unassigned variables left")

       # Random selection & removal of variable from _unassigned list
       idx = random.randrange(len(self._unassigned))
       decision_variable = self._unassigned.pop(idx)

       # Random decision for variable assignment: positive/negative
       return decision_variable if random.random() <= 0.5 else -decision_variable


    def select_decision_literal(self, heuristic: int) -> int:
     match heuristic:
        case 1:
            return self.vsids_heuristic()
        case 2:
            return self.most_recurring_heuristic()
        case 3:
            return self.random_heuristic()
        case _:
            raise ValueError(f"Unknown heuristic value: {heuristic}")

    def delete_learned_clauses_by_lbd(self, literal_block_dist_limit: float) -> None:
        
        literal_block_dist_limit = int(literal_block_dist_limit)
        new_learned_clauses = []
        for clause in self.learned_clauses:
            if clause.literal_block_distance > literal_block_dist_limit:
                self.watched_lists[abs(clause.literals[clause.watched_one])].remove(clause)
                if clause.watched_one != clause.watched_two:
                    self.watched_lists[abs(clause.literals[clause.watched_two])].remove(clause)

            else:
                new_learned_clauses.append(clause)

        self.learned_clauses = new_learned_clauses

    def reinstate(self) -> None:
        
        self.unit_clauses_queue.clear()
        self.backtrack(decision_level=0)

def cdcl(cnf_formula: CNF_Formula, heuristic: int = 1, conflicts_limit: int = 100,
         literal_block_dist_limit: float = 3.0) -> Tuple[bool, List[int], int, int, int]:

    decision_level = 0
    decisions = 0
    unit_propagations = 0
    reinstates = 0
    conflicts = 0

    # unit propagation: initial 
    propagated_literals, previous_of_conflict = cnf_formula.unit_propagation(decision_level)
    unit_propagations += len(propagated_literals)

    if previous_of_conflict:
        return False, [], decisions, unit_propagations, reinstates

    while not cnf_formula.all_assigned():
        # decision heuristic
        decision_literal = cnf_formula.select_decision_literal(heuristic)
        if decision_literal is None:
            break  # No unassigned literal found

        decision_level += 1

        # Partial assignment of formula with decision literal
        cnf_formula.literal_assignment(decision_literal, decision_level)
        decisions += 1

        # unit propagation: actual
        propagated_literals, previous_of_conflict = cnf_formula.unit_propagation(decision_level)
        unit_propagations += len(propagated_literals)

        while previous_of_conflict:
            conflicts += 1

            # perform reinstate and delete learned clauses with big LBD after checking the Conflict limit
            if conflicts >= conflicts_limit:
                conflicts = 0
                conflicts_limit = int(conflicts_limit * 1.1)
                literal_block_dist_limit *= 1.1
                reinstates += 1
                decision_level = 0
                cnf_formula.reinstate()
                cnf_formula.delete_learned_clauses_by_lbd(literal_block_dist_limit)
                break

            # Analyse conflict: learn new clause from the conflict and find out backtrack decision level
            backtrack_level = cnf_formula.conflict_inspection(previous_of_conflict, decision_level)
            if backtrack_level < 0:
                return False, [], decisions, unit_propagations, reinstates

            # Backtrack
            cnf_formula.backtrack(backtrack_level)
            decision_level = backtrack_level

            # Unit propagation of the learned clause
            propagated_literals, previous_of_conflict = cnf_formula.unit_propagation(decision_level)
            unit_propagations += len(propagated_literals)

    return True, list(cnf_formula.assignment_stack), decisions, unit_propagations, reinstates


def find_model(input_file: str, heuristic: int = 1, conflicts_limit: int = 100,
               literal_block_dist_limit: int = 3) -> Optional[Tuple[bool, list, float, int, int, int]]:

    if not input_file.endswith(".cnf"):
        print("Unsupported file extension. File extension must be `.cnf` for DIMACS.")
        return
# Exception handling for the input file
    try:
        with open(input_file, "r") as input:
            dimacs_formula = input.read().splitlines()
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    formula = [
        list(map(int, clause[:-2].strip().split()))
        for clause in dimacs_formula
        if clause and clause[0] not in {"c", "p", "%", "0"}
    ]

    cnf_formula = CNF_Formula(formula)

    start_time = time.time()
    sat, model, decisions, unit_propagations, reinstates = cdcl(cnf_formula, heuristic, conflicts_limit, literal_block_dist_limit)
    cpu_time = time.time() - start_time

    if sat:
        model.sort(key=abs)
        print("RESULT: SAT")
        assignment = {abs(literal): (literal > 0) for literal in model}
        print("ASSIGNMENT:", " ".join(f"{var}={int(val)}" for var, val in assignment.items()))
    else:
        print("RESULT: UNSAT")
    # print("Total CPU time =", cpu_time, "seconds")
    # print("Number of decisions =", decisions)
    # print("Number of steps of unit propagation =", unit_propagations)
    # print("Number of reinstates =", reinstates)
    return sat, model, cpu_time, decisions, unit_propagations, reinstates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDCL SAT Solver", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", type=str, help="Input `.cnf` file containing the formula.")
    parser.add_argument("--heuristic", type=int, default=1, help=(
            "Decision heuristic to use:\n"
            "  1 - VSIDS heuristic :- default\n"
            "  2 - Most occurrences in unsatisfied clauses\n"
            "  3 - Random unassigned literal"
        ))
    parser.add_argument("--conflicts_limit", type=int, default=100, help="Initial conflict limit before reinstating (default: 100)")
    parser.add_argument("--literal_block_dist_limit", type=int, default=3,help="Initial LBD limit for learned clause deletion (default: 3)")
    args = parser.parse_args()
    find_model(input_file=args.input, heuristic=args.heuristic, conflicts_limit=args.conflicts_limit, literal_block_dist_limit=args.literal_block_dist_limit)

