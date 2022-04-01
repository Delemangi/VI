import constraint

if __name__ == '__main__':
    problem = constraint.Problem()

    possible_meetings = [12, 13, 14, 15, 16, 17, 18, 19]
    simona_meetings = [13, 14, 16, 19]
    marija_meetings = [14, 15, 18]
    petar_meetings = [12, 13, 16, 17, 18, 19]

    participants = {
        'Marija_prisustvo': marija_meetings,
        'Simona_prisustvo': simona_meetings,
        'Petar_prisustvo': petar_meetings
    }

    problem.addVariable("Marija_prisustvo", constraint.Domain([0, 1]))
    problem.addVariable("Simona_prisustvo", constraint.Domain([0, 1]))
    problem.addVariable("Petar_prisustvo", constraint.Domain([0, 1]))
    problem.addVariable("vreme_sostanok", constraint.Domain(possible_meetings))

    problem.addConstraint(lambda x: x == 1, ['Simona_prisustvo'])
    for i in participants.keys():
        problem.addConstraint(lambda a, b, c=i: (a == 0 and b not in participants[c]) or (a == 1 and b in participants[c]), [i, 'vreme_sostanok'])

    [print(solution) for solution in problem.getSolutions()]
