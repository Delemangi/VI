import constraint


def attacking(q1, q2):
    return not (q1[0] == q2[0] or q1[1] == q2[1] or abs(q1[0] - q2[0]) == abs(q1[1] - q2[1]))


if __name__ == '__main__':
    problem = constraint.Problem()

    n = int(input())
    coordinates = [(i, j) for i in range(n) for j in range(n)]

    for variable in range(1, n + 1):
        problem.addVariable(variable, constraint.Domain(coordinates))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                problem.addConstraint(attacking, [i, j])

    if n <= 6:
        print(len(problem.getSolutions()))
    else:
        print(problem.getSolution())
