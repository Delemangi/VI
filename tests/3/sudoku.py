import constraint

if __name__ == '__main__':
    solver = input()

    if solver == 'BacktrackingSolver':
        problem = constraint.Problem()
    elif solver == 'RecursiveBacktrackingSolver':
        problem = constraint.Problem(constraint.RecursiveBacktrackingSolver())
    else:
        problem = constraint.Problem(constraint.MinConflictsSolver())

    variables = [(i, j) for i in range(9) for j in range(9)]

    for variable in variables:
        problem.addVariable(variable, constraint.Domain(set(range(1, 10))))

    for i in range(9):
        problem.addConstraint(constraint.AllDifferentConstraint(), set((i, j) for j in range(9)))
        problem.addConstraint(constraint.AllDifferentConstraint(), set((j, i) for j in range(9)))

    for i in range(3):
        for j in range(3):
            problem.addConstraint(constraint.AllDifferentConstraint(), set((i * 3 + k, j * 3 + l) for k in range(3) for l in range(3)))

    solution = problem.getSolution()

    if solution is None:
        print(None)
    else:
        print({(i * 9 + j): v for (i, j), v in solution.items()})
