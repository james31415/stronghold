import itertools
import numpy as np
import numpy.linalg as lin
import memoized

def unit(vec):
    return(vec / lin.norm(vec))

@memoized.memoized
def prod(points, first, second):
    if second < first:
        return(prod(points, second, first))
    unit1 = unit(points[first]["t"] - points[first]["p"])

    unit2 = unit(points[second]["t"] - points[second]["p"])

    return(np.dot(unit1, unit2))

def coef(points, i, j, n):
    if i == j:
        return(n - 1)
    else:
        return(-prod(points, i, j))

def solve_system(points):
    n = len(points)
    a = [[coef(points, i, j, n) for j in range(n)] for i in range(n)]
    p = [np.dot(unit(points[k]["t"] - points[k]["p"]), sum([points[i]["p"] for i in range(n)]) - n * points[k]["p"]) for k in range(n)]
    s = lin.solve(a, p)
    return(s)

def stronghold_sets(points, debug=False):
    n = len(points)
    l = []
    for x, y in itertools.combinations(range(n), 2):
        q = [points[x], points[y]]
        s = solve_system(q)
        if len(filter(lambda x: x < 0, s)) > 0:
            if p:
                print("({}, {}) point to different strongholds.".format(x, y))

            for i in range(len(l)):
                if x in l[i]:
                    break
            else:
                l = l + [set([x])]

            for i in range(len(l)):
                if y in l[i]:
                    break
            else:
                l = l + [set([y])]
        else:
            if p:
                print("({}, {}) point to same stronghold.".format(x, y))

            for i in range(len(l)):
                if x in l[i]:
                    l[i] = l[i] | set([y])
                    break
                elif y in l[i]:
                    l[i] = l[i] | set([x])
                    break
            else:
                l = l + [set((x, y))]
                
    return(l)

if __name__ == "__main__":
    points = [{"p": np.array([-223.457, 604.452]), "t": np.array([-233.518, 613.594])},
            {"p": np.array([-237.718, 547.065]), "t": np.array([-245.433, 558.430])},
            {"p": np.array([297.584, 746.493]), "t": np.array([287.264, 748.710])}]

    points = points + [{"p": np.array([-293.482, 279.496]), "t": np.array([-281.391, 277.522])}]
    n = len(points)

    l = stronghold_sets(points)
    print(l)

    for pq in l:
        if len(pq) == 1:
            continue

        print(pq)
        ppq = [points[j] for j in pq]
        s = solve_system(ppq)
        print("s: {}".format(s))
        pp = [ppq[i]["p"] + unit(ppq[i]["t"] - ppq[i]["p"])*s[i] for i in range(len(s))]
        ps = sum(pp)/n
        var = sum([np.dot(q - ps, q - ps) for q in pp])/n
        print("resultant points: {}".format(pp))
        print("average: {}".format(sum(pp)/n))
        print("deviation: {}".format(np.sqrt(var)))
