import constraint

if __name__ == '__main__':
    problem = constraint.Problem()
    variables = ["S", "E", "N", "D", "M", "O", "R", "Y", 'X1', 'X2', 'X3', 'X4']

    for variable in variables:
        problem.addVariable(variable, constraint.Domain(set(range(10))))

    problem.addConstraint(constraint.AllDifferentConstraint(), ["S", "E", "N", "D", "M", "O", "R", "Y"])
    problem.addConstraint(lambda a, b, c, d: a + b == c + 10 * d, ['D', 'E', 'Y', 'X1'])
    problem.addConstraint(lambda a, b, c, d, e: a + b + c == d + 10 * e, ['N', 'R', 'X1', 'E', 'X2'])
    problem.addConstraint(lambda a, b, c, d, e: a + b + c == d + 10 * e, ['E', 'O', 'X2', 'N', 'X3'])
    problem.addConstraint(lambda a, b, c, d, e: a + b + c == d + 10 * e, ['S', 'M', 'X3', 'O', 'X4'])
    problem.addConstraint(lambda a, b: a == b, ['X4', 'M'])
    # problem.addConstraint(lambda a: a != 0, ['S'])
    # problem.addConstraint(lambda a: a != 0, ['M'])

    solution = problem.getSolution()
    del solution['X1']
    del solution['X2']
    del solution['X3']
    del solution['X4']

    print(solution)
