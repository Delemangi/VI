import constraint

if __name__ == '__main__':
    problem = constraint.Problem(constraint.RecursiveBacktrackingSolver())
    members = {}
    leaders = {}

    members_num = int(input())
    for _ in range(members_num):
        member = input().split(' ')
        members[float(member[0])] = member[1]

    leaders_num = int(input())
    for _ in range(leaders_num):
        leader = input().split(' ')
        leaders[float(leader[0])] = leader[1]

    for i in range(5):
        problem.addVariable(str(i), constraint.Domain(members.keys()))
    problem.addVariable('L', constraint.Domain(leaders.keys()))

    problem.addConstraint(constraint.MaxSumConstraint(100))
    problem.addConstraint(constraint.AllDifferentConstraint())

    solutions = problem.getSolutions()
    solutions.sort(key=lambda x: sum(x.values()))

    solution = solutions[-1]

    print(f'Total score: {round(sum(v for _, v in solution.items()), 1)}')
    print(f'Team leader: {leaders[solution["L"]]}')
    for i in range(5):
        print(f'Participant {i + 1}: {members[solution[str(i)]]}')
