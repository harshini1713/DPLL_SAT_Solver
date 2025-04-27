import argparse
import random

import time
from typing import Tuple, Optional
from collections import deque
import numpy as np

class Clause:
    
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

    def partial_assignment_literals(self, assignment: list) -> list:
        
        unassigned = []
        for literal in self.literals:
            if assignment[abs(literal)] == literal:
                return []

            if assignment[abs(literal)] == 0:
                unassigned.append(literal)

        return list(unassigned)

    def watched_literal_update_pass(self, assignment: list, new_variable: int) -> Tuple[bool, int, Optional[int]]:
       
        #  the old watched literal index is `self.watched_one`
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

            # If the new watched literal index `self.watched_one` has not been found then the Clause is unit with
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
        self.formula = formula  # list of lists of literals
        self.clauses = [Clause(literals) for literals in self.formula]  # list of clauses
        self.learned_clauses = []
        self.variables = set()  # unordered unique set of variables in the formula
        self.watched_lists = {}  # dictionary: list of clauses with this `key` literal being watched
        self.unit_clauses_queue = deque()  # queue for unit clauses
        self.assignment_stack = deque()  # stack for representing the current assignment for backtracking
        self.assignment = None  # the assignment list with `variable` as index and `+variable/-variable/0` as values
        self.previous = None  # the previous list with `variable` as index and `Clause` as value
        self.decision_level = None  # the decision level list with `variable` as index and `decision level` as value
        self.positive_literal_counter = None
        self.negative_literal_counter = None

        for clause in self.clauses:
            # If the clause is unit right at the start, add it to the unit clauses queue
            if clause.watched_one == clause.watched_two:
                self.unit_clauses_queue.append((clause, clause.literals[clause.watched_two]))

            # For every literal in clause:
            for literal in clause.literals:
                variable = abs(literal)
                # - add variable to the set of all variables
                self.variables.add(variable)

                # - Create empty list of watched clauses for this variable, if it does not exist yet
                if variable not in self.watched_lists:
                    self.watched_lists[variable] = []

                # - Update the list of watched clauses for this variable
                if clause.literals[clause.watched_one] == literal or clause.literals[clause.watched_two] == literal:
                    if clause not in self.watched_lists[variable]:
                        self.watched_lists[variable].append(clause)

        # Set the assignment/previous/decision_level list of the Formula with initial values for each variable
        max_variable = max(self.variables)
        self.assignment = [0] * (max_variable + 1)
        self.previous = [None] * (max_variable + 1)
        self.decision_level = [-1] * (max_variable + 1)
        self.positive_literal_counter = np.zeros((max_variable + 1), dtype=np.float64)
        self.negative_literal_counter = np.zeros((max_variable + 1), dtype=np.float64)
        

    def literal_assignment(self, literal: int, decision_level: int) -> Tuple[bool, Optional[Clause]]:
       
        # Add literal to assignment stack and set the value of corresponding variable in the assignment list
        self.assignment_stack.append(literal)
        self.assignment[abs(literal)] = literal
        self.decision_level[abs(literal)] = decision_level

        # Copy the watched list of this literal because we need to delete some of the clauses from it during
        # iteration and that cannot be done while iterating through the same list
        watched_list = self.watched_lists[abs(literal)][:]

        # For every clause in the watched list of this variable perform the update of the watched literal and
        # find out which clauses become unit and which become unsatisfied in the current assignment
        for clause in watched_list:
            success, watched_literal, unit = clause.watched_literal_update_pass(self.assignment, abs(literal))

            # If the clause is not unsatisfied:
            if success:
                # If the watched literal was changed:
                if abs(watched_literal) != abs(literal):
                    # Add this clause to the watched list of the new watched literal
                    if clause not in self.watched_lists[abs(watched_literal)]:
                        self.watched_lists[abs(watched_literal)].append(clause)

                    # Remove this clause from the watched list of the old watched literal
                    self.watched_lists[abs(literal)].remove(clause)

                # If the clause is unit then add the clause to the unit clauses queue
                if unit:
                    if clause.literals[clause.watched_two] not in [x[1] for x in self.unit_clauses_queue]:
                        self.unit_clauses_queue.append((clause, clause.literals[clause.watched_two]))

            # If the clause is unsatisfied return False
            if not success:
                return False, clause

        return True, None

    def all_assigned(self) -> bool:
       
        return len(self.variables) == len(self.assignment_stack)

    def backtrack(self, decision_level: int) -> None:
       
        while self.assignment_stack and self.decision_level[abs(self.assignment_stack[-1])] > decision_level:
            literal = self.assignment_stack.pop()
            self.assignment[abs(literal)] = 0
            self.previous[abs(literal)] = None
            self.decision_level[abs(literal)] = -1

    @staticmethod
    def resolve(clause1: list, clause2: list, literal: int) -> list:
       
        in_clause1 = set(clause1)
        in_clause2 = set(clause2)
        in_clause1.remove(-literal)
        in_clause2.remove(literal)
        return list(in_clause1.union(in_clause2))

    def conflict_inspection(self, previous_of_conflict: Clause, decision_level: int) -> int:
       
        # If the conflict was detected at decision level 0, return -1
        if decision_level == 0:
            return -1

        # Find the literals of the assertive clause
        assertive_clause_literals = previous_of_conflict.literals
        current_assignment = deque(self.assignment_stack)
        while len([l for l in assertive_clause_literals if self.decision_level[abs(l)] == decision_level]) > 1:
            while True:
                literal = current_assignment.pop()
                if -literal in assertive_clause_literals:
                    assertive_clause_literals = self.resolve(assertive_clause_literals,
                                                             self.previous[abs(literal)].literals, literal)
                    break

     
        assertion_level = 0
        unit_literal = None
        watched_two = None
        decision_level_present = [False] * (decision_level + 1)
        for index, literal in enumerate(assertive_clause_literals):
            if assertion_level < self.decision_level[abs(literal)] < decision_level:
                assertion_level = self.decision_level[abs(literal)]

            if self.decision_level[abs(literal)] == decision_level:
                unit_literal = literal
                watched_two = index

            if not decision_level_present[self.decision_level[abs(literal)]]:
                decision_level_present[self.decision_level[abs(literal)]] = True

            self.positive_literal_counter = self.positive_literal_counter * 0.9
            self.negative_literal_counter = self.negative_literal_counter * 0.9
            if literal > 0:
                self.positive_literal_counter[literal] += 1

            else:
                self.negative_literal_counter[(abs(literal))] += 1

        # Find out LBD of the assertive clause
        literal_block_distance = sum(decision_level_present)

        # Find the `watched_one` index for the assertive clause which is the index of the last assigned literal
        # in the assertive clause with decision level equal to the assertion level
        watched_one = None
        if len(assertive_clause_literals) > 1:
            current_assignment = deque(self.assignment_stack)
            found = False
            while current_assignment:
                literal = current_assignment.pop()
                if self.decision_level[abs(literal)] == assertion_level:
                    for index, clause_literal in enumerate(assertive_clause_literals):
                        if abs(literal) == abs(clause_literal):
                            watched_one = index
                            found = True
                            break

                if found:
                    break

        else:
            watched_one = watched_two

        # Create the assertive clause and update the watched lists of the watched literals
        assertive_clause = Clause(assertive_clause_literals, watched_one=watched_one, watched_two=watched_two, learned=True, literal_block_distance=literal_block_distance)
        self.watched_lists[abs(assertive_clause.literals[assertive_clause.watched_one])].append(assertive_clause)
        if assertive_clause.watched_one != assertive_clause.watched_two:
            self.watched_lists[abs(assertive_clause.literals[assertive_clause.watched_two])].append(assertive_clause)

        # Add the assertive clause into the list of learned clauses
        self.learned_clauses.append(assertive_clause)

        # Clear the unit clauses queue and add the assertive clause into the unit clauses queue
        # together with its unit literal
        self.unit_clauses_queue.clear()
        self.unit_clauses_queue.append((assertive_clause, unit_literal))

        return assertion_level

    def unit_propagation(self, decision_level: int) -> Tuple[list, Optional[Clause]]:
        
        propagated_literals = []
        while self.unit_clauses_queue:
            unit_clause, unit_clause_literal = self.unit_clauses_queue.popleft()
            propagated_literals.append(unit_clause_literal)
            self.previous[abs(unit_clause_literal)] = unit_clause

            success, previous_of_conflict = self.literal_assignment(unit_clause_literal, decision_level)
            if not success:
                return propagated_literals, previous_of_conflict

        return propagated_literals, None

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

    def vsids_heuristic(self) -> int:
        
        decision_literal = None
        best_counter = 0
        for variable in self.variables:
            if self.assignment[variable] == 0:
                if self.positive_literal_counter[variable] > best_counter:
                    decision_literal = variable
                    best_counter = self.positive_literal_counter[variable]

                if self.negative_literal_counter[variable] >= best_counter:
                    decision_literal = -variable
                    best_counter = self.negative_literal_counter[variable]

        return decision_literal
        
    def random_heuristic(self) -> int:
        
        if not hasattr(self, '_unassigned') or not self._unassigned:
          self._unassigned = [var for var in self.variables if self.assignment[var] == 0]

        if not self._unassigned:
           raise Exception("No unassigned variables left")

    # Randomly select and remove a variable from the unassigned list
        idx = random.randrange(len(self._unassigned))
        decision_variable = self._unassigned.pop(idx)

    # Randomly decide whether to assign positive or negative
        return decision_variable if random.random() <= 0.5 else -decision_variable

    def select_decision_literal(self, heuristic: int) -> int:
     match heuristic:
        case 0:
            return self.most_recurring_heuristic()
        case 1:
            return self.vsids_heuristic()
        case 2:
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
         literal_block_dist_limit: int = 3) -> Tuple[bool, int, int, int]:
   
    # Counters for number of decisions, unit propagations
    decision_level = 0
    decisions = 0
    unit_propagations = 0
    reinstates = 0
    conflicts = 0

    # Unit propagation
    propagated_literals, previous_of_conflict = cnf_formula.unit_propagation(decision_level)
    unit_propagations += len(propagated_literals)

    if previous_of_conflict:
        return False, [], decisions, unit_propagations, reinstates

    
    while not cnf_formula.all_assigned():
        # Find the literal for decision by finding one using decision heuristic
        
        decision_literal = cnf_formula.select_decision_literal(heuristic)

        decision_level += 1

        # Perform the partial assignment of the formula with the decision literal
        cnf_formula.literal_assignment(decision_literal, decision_level)
        decisions += 1

        # Unit propagation
        propagated_literals, previous_of_conflict = cnf_formula.unit_propagation(decision_level)
        unit_propagations += len(propagated_literals)

        while previous_of_conflict:
            conflicts += 1

            # If the amount of conflicts reached the limit, perform reinstate and delete learned clauses with big LBD
            if conflicts == conflicts_limit:
                conflicts = 0
                conflicts_limit = int(conflicts_limit * 1.1)
                literal_block_dist_limit = literal_block_dist_limit * 1.1
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


def find_model(input_file: str,  heuristic: int = 1, conflicts_limit: int = 100,
               literal_block_dist_limit: int = 3) -> Optional[Tuple[bool, float, int, int, int]]:
   
    
    if input_file[-3:] == "cnf":
        input = open(input_file, mode="r")

    else:
        print("Unsupported file extension. File extension must be `.cnf` for DIMACS, or `.sat` for the simplified "
              "SMT-LIB format.")
        return

    dimacs_formula = input.read()
    dimacs_formula = dimacs_formula.splitlines()

    formula = [list(map(int, clause[:-2].strip().split())) for clause in dimacs_formula if clause != "" and
               clause[0] not in ["c", "p", "%", "0"]]

    cnf_formula = CNF_Formula(formula)
    start_time = time.time()
    sat, model, decisions, unit_propagations, reinstates = cdcl(cnf_formula, heuristic, conflicts_limit,
                                                              literal_block_dist_limit)
    cpu_time = time.time() - start_time
    if sat:
        model.sort(key=abs)
        print("RESULT: SAT")
        #print("Model =", model)
        assignment = {abs(literal): (literal > 0) for literal in model}
        print("ASSIGNMENT:", " ".join(f"{var}={int(val)}" for var, val in assignment.items()))
        #print("Possible missing literals can have arbitrary value.")

    else:
        print("RESULT: UNSAT")

    #print("Total execution time =", cpu_time*1000, "seconds")
    #print("Number of decisions =", decisions)
    #print("Number of steps of unit propagation =", unit_propagations)
    #print("Number of reinstates =", reinstates)

    return sat, model, cpu_time, decisions, unit_propagations, reinstates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file which contains a description of a formula.")
    parser.add_argument("--heuristic", type=int, default=1, help="Specify a decision heuristic: `0` - pick the "
                                                                 "unassigned literal which occurs in the largest "
                                                                 "number of not satisfied clauses, `1` - pick the "
                                                                 "unassigned literal based on VSIDS heuristic, "
                                                                 "`2` - pick the random unassigned literal")
    parser.add_argument("--conflicts_limit", default=100, help="The initial limit on the number of conflicts before "
                                                               "the CDCL solver reinstates")
    parser.add_argument("--literal_block_dist_limit", default=3, help="The initial limit on the number of different decision levels "
                                                       "in the learned clause for clause deletion")
    args = parser.parse_args()

    find_model(args.input, args.heuristic, args.conflicts_limit, args.literal_block_dist_limit)

