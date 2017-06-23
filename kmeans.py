import Image
import numpy as np
import matplotlib.pyplot as plt


def kmeans(points, k):
    n = points.shape[0]
    # initialize centers
    centers = [[] for x in xrange(k)]
    for i in xrange(k):
        if k == 2:
            step = 50
        elif k == 8:
            step = 4
        else:
            step = 2
        centers[i] = [step * i] * 3
    centers = np.array(centers)

    variation = 100
    while variation > 2:
        assign = [[] for x in xrange(k)]
        # assign points to centers
        Asq = points ** 2
        Asq = np.sum(Asq, axis=1)
        Asq = np.reshape(Asq, (n, 1))
        Asq = np.tile(Asq, k)
        Bsq = centers ** 2
        Bsq = np.sum(Bsq, axis=1)
        Bsq = np.reshape(Bsq, (k, 1))
        Bsq = np.tile(Bsq, n)
        Bsq = Bsq.T
        cross = points.dot(centers.T)
        Dis = Asq + Bsq - 2 * cross
        C = np.argmin(Dis, 1)
        for i in xrange(n):
            c = C[i]
            assign[c].append(i)

        # adjust centers
        new_centers = np.zeros((k, 3), dtype=np.int64)
        for i in xrange(k):
            if len(assign[i]) > 0:
                s = np.zeros(3)
                for x in assign[i]:
                    s += points[x, :]
                new_centers[i, :] = s / len(assign[i])
        variation = np.max(np.abs(new_centers - centers))
        centers = new_centers

    return centers, C

img = Image.open("Sea.jpg")
mat = np.array(img, dtype=np.int64)
m = mat.shape[0]
n = mat.shape[1]
pixels = np.zeros((m * n, 3))
for i in xrange(m):
    for j in xrange(n):
        pixels[i * n + j] = mat[i, j]

for k in [2, 8, 16, 64]:
    (centers, belonging) = kmeans(pixels, k)
    for i in xrange(m):
        for j in xrange(n):
            c = belonging[i * n + j]
            mat[i, j] = centers[c, :]
    mat = np.uint8(mat)
    res_img = Image.fromarray(mat)
    res_img.save("Sea_{}.jpg".format(k))
