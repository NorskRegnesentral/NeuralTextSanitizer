# install google-ortools
# python -m pip install --upgrade --user ortools

import numpy as np
from ortools.sat.python import cp_model

class linearOpt:
    def __init__(self, n, m, constraints, scores, target="min"):
        """
        :param n: How many entities
        :param m: How many options do we have, e.g [MASK Original, KEEP Original, Change1] -> m = 3
        :param constraints: [(entity i1, option j1), (entity i2, option j2)] should not appear together
        e.g. [(0,1)] = We can not choose option 1 [KEEP] for entity 0.
        e.g. [(0,2),(1,3)] We can not choose entity0 option2 and entity1 option3 together.
        :param scores: The score for each entity's options, ndarray of shape = (n,m)
        e.g. scores[i,j] = the score for choosing entity i option j.

        In the case of only considering KEEP and MASK, m = 2, and if we use option0 for KEEP, option1 for MASK
        constraints will have form like [(...,1)]
        or [(...,1),(...,1)]
        """
        self.model = cp_model.CpModel()

        # Create Variables
        self.choices = {}
        for i in range(n):
            for j in range(m):
                self.choices[i, j] = self.model.NewBoolVar(f'entity{i}-option{j}')

        # Constraint:
        ## 1. Each entity should choose only 1 option
        for i in range(n):
            self.model.AddExactlyOne(self.choices[i, j] for j in range(m))
        ## 2. Blacklist: For each k-entities pair, we can only keep at most k-1 entities.
        for pair in constraints:
            self.model.Add(sum(self.choices[choice] for choice in pair) <= len(pair) - 1)

        if target == "min":
            self.model.Maximize(- sum(scores[i, j] * self.choices[i, j] for i in range(n) for j in range(m)))
        else:
            self.model.Maximize(sum(scores[i, j] * self.choices[i, j] for i in range(n) for j in range(m)))

    def solve(self):
        """
        :return: the decision for each entity
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print('Solution found, value is ', solver.ObjectiveValue())
            if status == cp_model.OPTIMAL:
                print("Optimal Solution")
            elif status == cp_model.FEASIBLE:
                print("Feasible Solution")

            # Print Solutions
            res = []
            for choice, var in self.choices.items():
                if solver.Value(var) == 1:
                    print(f'Entity {choice[0]} choose option {choice[1]}')
                    res.append(choice[1])
            return res
        else:
            print('No solution found.')


if __name__ == "__main__":
    n = 5
    m = 2
    constraints = [
        [(4,1)],                # Can't keep entity 4
        [(2,1),(3,1)],          # entity2 and entity3 can not appear together
        [(0,1),(1,1),(2,1)],    # entity0, entity1, entity2 can not appear together
    ]
    scores = np.array([
        [10, 8],
        [7, 5],
        [6, 3],
        [5, 4],
        [7, 5],
    ])

    tmp = linearOpt(n, m, constraints, scores, target="min")
    res = tmp.solve()
    print(res)
