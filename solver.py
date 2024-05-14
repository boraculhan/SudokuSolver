"""

This code is a modified version of the Sudoku IP formulation example 
provided in the documentation of PuLP library.

Source: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html

"""

from pulp import LpProblem, LpVariable, lpSum, value, LpStatus, PULP_CBC_CMD
import numpy as np

class SolveSudoku():
    """
    
    Usage: Input data must be a list of triplets in (k, i, j) form.
    k := The clue number
    i := The row of the clue number k
    j := The column of the clue number k
    
    """
    def __init__(self, input_data):
        self.input_data = input_data
        # All rows, columns and values within a Sudoku take values from 1 to 9
        self.VALS = self.ROWS = self.COLS = range(1, 10)
    
        # The boxes list is created, with the row and column index of each square in each box
        self.Boxes = [
            [(3 * i + k + 1, 3 * j + l + 1) for k in range(3) for l in range(3)]
            for i in range(3)
            for j in range(3)
            ]

        self.prob = LpProblem("SudokuProblem")

        # The decision variables are created
        self.choices = LpVariable.dicts("Choice", (self.VALS, self.ROWS, self.COLS), cat="Binary")
        self.output_data = np.zeros((9, 9))

    def BuildConstraints(self):
        # A constraint ensuring that only one value can be in each square is created
        for r in self.ROWS:
            for c in self.COLS:
                self.prob += lpSum([self.choices[v][r][c] for v in self.VALS]) == 1
        
        # The row, column and box constraints are added for each value
        for v in self.VALS:
            for r in self.ROWS:
                self.prob += lpSum([self.choices[v][r][c] for c in self.COLS]) == 1
        
            for c in self.COLS:
                self.prob += lpSum([self.choices[v][r][c] for r in self.ROWS]) == 1
        
            for b in self.Boxes:
                self.prob += lpSum([self.choices[v][r][c] for (r, c) in b]) == 1
        
        for v, r, c in self.input_data:
            self.prob += self.choices[v][r][c] == 1

    def SolveLP(self):
        self.prob.solve(PULP_CBC_CMD(msg=False))        
        
        for r in self.ROWS:
            for c in self.COLS:
                for v in self.VALS:
                    if value(self.choices[v][r][c]) == 1:
                        self.output_data[r-1, c-1] = v
                        
        return LpStatus[self.prob.status], self.output_data