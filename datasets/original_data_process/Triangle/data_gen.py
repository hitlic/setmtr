import random as rand
import math


def triangle_check(p1, p2, p3):
    if sum(p1) ==0 or sum(p2) ==0 or sum(p2) == 0:
        return False
    def dist(x, y):
        return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    ds = sorted([dist(p1, p2), dist(p1, p3), dist(p2, p3)])
    return ds[0] + ds[1] > ds[2] + 0.01  # 三点落在一线上的判断不可能完全精确，也许用于控制难易程度


def triangle_gen(num, x_span=(0, 1), y_span=(0, 1)):
    triangles = []
    n = 0
    while n < num:
        ps = [[rand.random()*(x_span[1] - x_span[0]) + x_span[0],
               rand.random()*(y_span[1] - y_span[0]) + y_span[0]] for _ in range(3)]
        if triangle_check(*ps):
            n += 1
        triangles.append(ps)
    return triangles

triangles = triangle_gen(20000)

with open("triangle.txt", 'w', encoding="utf-8") as f:
    for ps in triangles:
        f.write(f"@[{' '.join(map(str, ps[0]))}],@[{' '.join(map(str, ps[1]))}],@[{' '.join(map(str, ps[2]))}]\n")

from matplotlib import pyplot as plt
x = []
y = []
for ps in triangles:
    x.extend([ps[0][0], ps[1][0], ps[2][0]])
    y.extend([ps[0][1], ps[1][1], ps[2][1]])
plt.scatter(x, y)
plt.show()
