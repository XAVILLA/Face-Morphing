import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.spatial import Delaunay
from scipy.spatial import tsearch
from matplotlib.path import Path

from scipy import misc
import numpy as np
import ast
import copy
import time
import os

POINT_COUNT = 60
FRAME_COUNT = 45
CORNER_POINTS = [[0, 0], [0, 499], [499, 0], [499, 499]]


def findPoints(imA):
    plt.imshow(imA)
    i = 0
    imA_points = []
    while i < POINT_COUNT:
        x = plt.ginput(1, timeout=0)
        imA_points.append([x[0][0], x[0][1]])
        plt.scatter(x[0][0], x[0][1])
        plt.draw()
        i += 1
    plt.close()
    for corner in CORNER_POINTS:
        imA_points.append(corner)
    return imA_points


def weightsPerFrame():
    return np.linspace(0.0, 1.0, FRAME_COUNT)


def findAverageShape(imA_points, imB_points, weight):
    shape = []
    for index in range(len(imA_points)):
        pointA = imA_points[index]
        pointB = imB_points[index]
        x = weight * pointA[0] + (1 - weight) * pointB[0]
        y = weight * pointA[1] + (1 - weight) * pointB[1]
        shape.append([x, y])

    return np.array(shape)


def findCorrespondences(imA, imB, filename):
    imA_points = findPoints(imA)
    imB_points = findPoints(imB)

    f = open("./stored_{}_points.py".format(filename), "w")
    f.write("imA_points = " + str(imA_points) + "\n")
    f.write("imB_points = " + str(imB_points) + "\n")
    f.close()
    return imA_points, imB_points


def findAffine(triMid, im_points, mid_points):
    affineMatrices = []
    for tri in triMid.simplices:
        src = im_points[tri,]
        dest = mid_points[tri,]
        affineMatrices.append(computeAffine(src, dest))
    return affineMatrices


def computeAffine(tri1, tri2):
    A = np.matrix("{} {} 1 0 0 0;".format(tri1[0][0], tri1[0][1])()
                  + "0 0 0 {} {} 1;".format(tri1[0][0], tri1[0][1])
                  + "{} {} 1 0 0 0;".format(tri1[1][0], tri1[1][1])
                  + "0 0 0 {} {} 1;".format(tri1[1][0], tri1[1][1])
                  + "{} {} 1 0 0 0;".format(tri1[2][0], tri1[2][1])
                  + "0 0 0 {} {} 1".format(tri1[2][0], tri1[2][1]))

    b = np.matrix("{} {} {} {} {} {}".format(tri2[0][0], tri2[0][1], tri2[1][0], tri2[1][1], tri2[2][0], tri2[2][1]))

    Affine_values = np.linalg.lstsq(A, np.transpose(b))[0]
    affine_Matrix = np.vstack((np.reshape(Affine_values, (2, 3)), [0, 0, 1]))

    return affine_Matrix


def findMidWayFace(imA, imB, imA_points, imB_points, imMid_points, triMid, weight):
    affineAMatrices = findAffine(triMid, imA_points, imMid_points)
    affineBMatrices = findAffine(triMid, imB_points, imMid_points)
    morphedIm = np.zeros(imA.shape)

    for y in range(imA.shape[0]):
        for x in range(imA.shape[1]):
            tri_index = tsearch(triMid, (x, y))
            affined_pointA = np.dot(np.linalg.inv(affineAMatrices[tri_index]), [x, y, 1])
            affined_pointB = np.dot(np.linalg.inv(affineBMatrices[tri_index]), [x, y, 1])

            affineAx = np.int(affined_pointA[0, 0])
            affineAy = np.int(affined_pointA[0, 1])
            affineBx = np.int(affined_pointB[0, 0])
            affineBy = np.int(affined_pointB[0, 1])

            morphedIm[y, x, :] = imA[affineAy, affineAx, :] * weight + imB[affineBy, affineBx, :] * (1 - weight)

    return morphedIm


def morph(imA, imB, imA_points, imB_points):
    weights = weightsPerFrame()
    print("begin")
    triA = Delaunay(imA_points)
    print("triA")
    triB = Delaunay(imB_points)
    print("triB")

    for i in range(FRAME_COUNT):
        print(i)
        mid_Points = findAverageShape(imA_points, imB_points, weights[i])
        triMid = Delaunay(mid_Points)
        frame = findMidWayFace(imA, imB, imA_points, imB_points, mid_Points, triMid, triA, triB, weights[i])
        misc.imsave("mortyframe_{}.jpg".format(i), frame)


def overlay(im, im_points):
    im_points = np.array(im_points)
    tri = Delaunay(im_points)
    plt.triplot(im_points[:, 0], im_points[:, 1], tri.simplices.copy())
    plt.plot(im_points[:, 0], im_points[:, 1], 'o')
    plt.imshow(im)
    plt.show()


def findMeanFace(directory):
    cwd = os.getcwd()
    face_directory = os.path.join(cwd, directory)
    im_points = []
    im_names = []
    for filename in os.listdir(face_directory):
        name = os.path.splitext(filename)
        if name[1] == ".asf":
            points_path = os.path.join(face_directory, filename)
            im_path = os.path.join(face_directory, name[0] + ".bmp")
            pointsMatrix = parse_files(points_path, im_path)
            im_points.append(pointsMatrix)
            im_names.append(im_path)

    sumMatrix = np.array(sum(im_points))
    average_points = sumMatrix / len(im_points)
    f = open("./dane_points.py", "w")
    f.write("average_points = " + str(average_points.tolist()) + "\n")
    f.close()
    warpMeanFace(im_names, im_points, average_points)


def parse_files(points_path, im_path):
    image = plt.imread(im_path)
    height = image.shape[0]
    width = image.shape[1]

    with open(points_path, 'r') as file:
        lines = file.readlines()
        lines = [line.split() for line in lines if line and line[0] in "0123456"]
        del lines[0], lines[len(lines) - 1]
        matrix = []
        for line in lines:
            x = np.int(width * float(line[2]))
            y = np.int(height * float(line[3]))
            matrix.append([x, y])
    matrix.append([0, 0])
    matrix.append([width, 0])
    matrix.append([0, height])
    matrix.append([width, height])
    return np.matrix(matrix)


def warpMeanFace(im_names, im_points, average_points, saved_name, t=1):
    im_points = np.array(im_points)
    triAverage = Delaunay(average_points)
    image = plt.imread(im_names[0])
    height = image.shape[0]
    width = image.shape[1]

    morphedIm = np.zeros((height, width, 3), dtype="float32")
    for i in range(len(im_names)):
        image = plt.imread(im_names[i])
        affineMatrices = findAffine(triAverage, im_points[i], average_points)
        for y in range(height):
            for x in range(width):
                tri_index = tsearch(triAverage, (x, y))
                affined_point = np.dot(np.linalg.inv(affineMatrices[tri_index]), [x, y, 1])

                affineX = affined_point[0, 0]
                affineY = affined_point[0, 1]

                morphedIm[y, x, :] += image[np.int(affineY), np.int(affineX), :]
    morphedIm = morphedIm / len(im_names)
    misc.imsave(saved_name, morphedIm)


def main():
    ### Uncomment this section to apply morphing
    imA = plt.imread("kai_start.jpg")
    imB = plt.imread("george_small.jpg")

    imA_points, imB_points = findCorrespondences(imA, imB, "kai")


# morphed_im = morph(imA, imB, np.array(imA_points), np.array(imB_points))

### Uncomment this section to apply the mean face
# findMeanFace("data")

### Uncomment this section to retrieve average faces from the Dane Set.
# import dane_points
# average_points = dane_points.average_points
# fileA = "./data/27-1m"
# warpMeanFace([fileA + ".bmp"], [np.array(parse_files(fileA + ".asf", fileA + ".bmp"))], np.array(average_points), "dane3_morph.jpg")

### Uncomment this section to retrieve the morph of my face to the average and vice versa.
# import stored_average_points
# imA_points = stored_average_points.imB_points
# imB_points = stored_average_points.imA_points
# warpMeanFace(["average_face_resized.jpg"], [imA_points], np.array(imB_points), "dane_kai.jpg")

### Uncomment this section to do caricatures.
# import stored_average_points
# imA_points = np.array(stored_average_points.imA_points)
# imB_points = np.array(stored_average_points.imB_points)
# t = 0

# imC_points = (imA_points - imB_points) * t + imB_points
# warpMeanFace(["kai_white.jpg"], [imA_points], np.array(imC_points), "car3_kai.jpg")


main()