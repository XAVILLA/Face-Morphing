import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import skimage.transform as tr
from skimage.draw import polygon
import sys
import numpy as np
import skimage.io as io
import cv2
import os.path as path
import os
import time

POINT_COUNT = 44
FRAME_NUM = 46


def computeAffine(tri1, tri2):
    A = np.matrix([[tri1[0][0], tri1[0][1], 1, 0, 0, 0],
                   [0, 0, 0, tri1[0][0], tri1[0][1], 1],
                   [tri1[1][0], tri1[1][1], 1, 0, 0, 0],
                   [0, 0, 0, tri1[1][0], tri1[1][1], 1],
                   [tri1[2][0], tri1[2][1], 1, 0, 0, 0],
                   [0, 0, 0, tri1[2][0], tri1[2][1], 1]])

    b = np.matrix([[tri2[0][0], tri2[0][1], tri2[1][0], tri2[1][1], tri2[2][0], tri2[2][1]]])

    Affine_values = np.linalg.lstsq(A, np.transpose(b))[0]
    reshape = np.reshape(Affine_values, (2, 3))
    affine = np.vstack((reshape, [0, 0, 1]))

    return affine


def findAffine(triMid, im_points, mid_points):
    affineMatrices = []
    for tri in triMid.simplices:
        # print(im_points.shape)
        src = [im_points[tri[0]], im_points[tri[1]], im_points[tri[2]]]
        dest = [mid_points[tri[0]], mid_points[tri[1]], mid_points[tri[2]]]
        affineMatrices.append(computeAffine(src, dest))
    return affineMatrices


def create_video(folder):
    img_array = []
    height, width, layers = 0, 0, 0
    for i in range(FRAME_NUM):
        name = folder + "/" + str(i) + ".jpg"
        img = cv2.imread(name)
        height, width, layers = img.shape

        img_array.append(img)

    video = cv2.VideoWriter(folder + "/sample.avi", 0, 30, (width, height))

    for image in img_array:
        video.write(image)

    video.release()


def scaleData(refImg, img2, img2points):
    hR, wR = refImg.shape[:2]
    h, w = img2.shape[:2]
    new_data = []
    for point in img2points:
        scaled = [point[0] * wR / w, point[1] * hR / h]
        new_data.append(scaled)

    new_img2 = tr.resize(img2, refImg.shape)

    return new_img2, new_data


def findPoints(img):
    print(img.shape)
    plt.imshow(img)
    i = 0
    img_points = []
    while i < POINT_COUNT:
        x = plt.ginput(1, timeout=0)
        print(x)
        img_points.append([x[0][0], x[0][1]])
        plt.scatter(x[0][0], x[0][1])
        plt.draw()
        i += 1
    plt.close()
    h, w = img.shape[:2]
    corners = [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]
    for c in corners:
        img_points.append(c)
    assert len(img_points) == POINT_COUNT + 4
    return img_points


def weightsPerFrame(num):
    return np.linspace(0.0, 1.0, num)


def storePoints(im, name):
    im_points = findPoints(im)

    f = open("points_data/{}_data.txt".format(name), "w")
    f.write("im_points = " + str(im_points) + "\n")
    f.close()
    return im_points


def process_data(name):
    filename = "points_data/{}_data.txt".format(name)
    with open(filename) as file:
        data = file.readlines()[0]
    data = data[13:][:-2]
    data = data + ","
    data = data.replace(" ", "").split("[")[1:]
    data = [a.replace("],", "") for a in data]
    data = [a.split(",") for a in data]
    data = [[float(a[0]), float(a[1])] for a in data]
    print(len(data), "data points read from file")
    data = np.array(data)
    assert len(data) == POINT_COUNT + 4
    return data


def wrapInto(fromImg, fromData, toImg, toData, alpha):
    shapeDiff = toData - fromData
    newShape = fromData + alpha*shapeDiff
    triMid = Delaunay(newShape)
    affineMatrices = findAffine(triMid, fromData, newShape)

    produced_image = np.zeros(fromImg.shape)

    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]
        affineAM = affineMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        Ymax, Xmax, layer = produced_image.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)

        produced_image[morphYs, morphXs] = fromImg[imAYs, imAXs]

    return produced_image


def bair():
    data_num = 48
    imgs = []
    points = []
    for i in range(data_num):
        dataname = "bair" + str(i)
        imgname = "MeanFace/Baircut/BAIR" + str(i) + ".jpg"
        data = process_data(dataname)
        img = io.imread(imgname)
        assert len(data) == 48
        # print(img.shape)
        assert img.shape == (600, 450, 3)
        points.append(data)
        imgs.append(img)
    shape_tracker = np.zeros(points[0].shape)
    for s in points:
        shape_tracker += s
    mean_shape = shape_tracker / data_num
    name = "bair_mean"
    f = open("points_data/{}_data.txt".format(name), "w")
    store_mean = []
    for a in mean_shape:
        store_mean.append([a[0], a[1]])
    f.write("im_points = " + str(store_mean))
    f.close()
    triMid = Delaunay(mean_shape)
    result_tracker = []
    for i in range(data_num):
        img = imgs[i]
        point = points[i]
        # print(point.shape)
        affineAMatrices = findAffine(triMid, point, mean_shape)
        produced_img = np.zeros(img.shape)

        for i in range(len(triMid.simplices)):
            tri = triMid.simplices[i]
            x1, y1 = triMid.points[tri[0]]
            x2, y2 = triMid.points[tri[1]]
            x3, y3 = triMid.points[tri[2]]

            affineAM = affineAMatrices[i]

            morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
            morphXs = morphedPoly[0, :]
            morphYs = morphedPoly[1, :]

            morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

            imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
            imApolyInt = imApoly.astype(np.uint32)[:2, :]

            imAXs = imApolyInt[0, :]
            imAYs = imApolyInt[1, :]

            Ymax, Xmax, layer = img.shape

            if not np.all(imAXs < Xmax):
                imAXs = imAXs - (imAXs >= Xmax)
            if not np.all(imAYs < Ymax):
                imAYs = imAYs - (imAYs >= Ymax)
            # print(img.shape)

            produced_img[morphYs, morphXs] = img[imAYs, imAXs]
        result_tracker.append(produced_img)
    result_img = np.zeros(imgs[0].shape)

    for im in result_tracker:
        result_img += im
    result_img /= data_num

    io.imsave("result.jpg", result_img)





def findAverageShape(imA_points, imB_points, weight):
    shape = []
    for index in range(len(imA_points)):
        pointA = imA_points[index]
        pointB = imB_points[index]
        x = weight * pointB[0] + (1 - weight) * pointA[0]
        y = weight * pointB[1] + (1 - weight) * pointA[1]
        shape.append([x, y])

    return np.array(shape)


def findMidWayFace(imA, imB, imA_points, imB_points, imMid_points, triMid, weight):
    affineAMatrices = findAffine(triMid, imA_points, imMid_points)
    affineBMatrices = findAffine(triMid, imB_points, imMid_points)
    morphedIm = np.zeros(imA.shape)

    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]

        affineAM = affineAMatrices[i]
        affineBM = affineBMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis = 0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imBpoly = np.dot(np.linalg.inv(affineBM), morphedPolyScale)
        imBpolyInt = imBpoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        imBXs = imBpolyInt[0, :]
        imBYs = imBpolyInt[1, :]

        Ymax, Xmax, layer = imA.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)
        if not np.all(imBXs < Xmax):
            imBXs = imBXs - (imBXs >= Xmax)
        if not np.all(imBYs < Ymax):
            imBYs = imBYs - (imBYs >= Ymax)

        morphedIm[morphYs, morphXs] = imA[imAYs, imAXs]*(1-weight) + imB[imBYs, imBXs]*weight

    return morphedIm


def morph(imA, imB, imA_points, imB_points, folder, framenum):
    weights = weightsPerFrame(framenum)

    if not path.exists(folder):
        os.mkdir(folder)
    io.imsave(folder + "/" + "0.jpg", imA.astype(np.uint8))

    for i in range(1, framenum - 1):
        starttime = time.time()

        print("Producing frame number ", i)

        mid_Points = findAverageShape(imA_points, imB_points, weights[i])

        triMid = Delaunay(mid_Points)

        frame = findMidWayFace(imA, imB, imA_points, imB_points, mid_Points, triMid, weights[i]).astype(np.uint8)
        print("--- %s seconds ---" % (time.time() - starttime))

        io.imsave(folder + "/{}.jpg".format(i), frame)
    print(np.max(imB.astype(np.uint8)))
    io.imsave(folder + "/" + str(framenum - 1) + ".jpg", imB.astype(np.uint8))


def femaleShape():
    zzxdata = process_data("zzx")
    zzxImg = io.imread("images/zzx.jpg")
    maledata = process_data("male")
    femaledata = process_data("female")
    maleImg = io.imread("images/male.jpg")
    femaleImg = io.imread("images/female.jpg")

    maleImgN, maledataN = scaleData(zzxImg, maleImg, maledata)
    femaleImgN, femaledataN = scaleData(zzxImg, femaleImg, femaledata)

    zzxdata, maledataN, femaledataN = np.array(zzxdata), np.array(maledataN), np.array(femaledataN)

    shape_diff = femaledataN - maledataN
    newShape = zzxdata + shape_diff*0.5

    triMid = Delaunay(newShape)
    affineMatrices = findAffine(triMid, zzxdata, newShape)
    producedImg = np.zeros(zzxImg.shape)

    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]
        affineAM = affineMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        Ymax, Xmax, layer = producedImg.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)

        producedImg[morphYs, morphXs] = zzxImg[imAYs, imAXs]

    return producedImg

def femaleColor():
    zzxdata = process_data("zzx")
    zzxImg = io.imread("images/zzx.jpg")
    maledata = process_data("male")
    femaledata = process_data("female")
    maleImg = io.imread("images/male.jpg")
    femaleImg = io.imread("images/female.jpg")

    maleImgN, maledataN = scaleData(zzxImg, maleImg, maledata)
    femaleImgN, femaledataN = scaleData(zzxImg, femaleImg, femaledata)

    zzxdata, maledataN, femaledataN = np.array(zzxdata), np.array(maledataN), np.array(femaledataN)

    maleInFemaleShape = wrapInto(maleImgN, maledataN, femaleImgN, femaledataN, 1)

    color_diff = femaleImgN - maleInFemaleShape

    producedImg = np.zeros(zzxImg.shape)

    triMid = Delaunay(zzxdata)
    affineMatrices = findAffine(triMid, femaledataN, zzxdata)


    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]
        affineAM = affineMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        Ymax, Xmax, layer = producedImg.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)

        producedImg[morphYs, morphXs] = zzxImg[morphYs, morphXs] + color_diff[imAYs, imAXs]*50

    return producedImg

def femaleBoth():
    zzxdata = process_data("zzx")
    zzxImg = io.imread("images/zzx.jpg")
    maledata = process_data("male")
    femaledata = process_data("female")
    maleImg = io.imread("images/male.jpg")
    femaleImg = io.imread("images/female.jpg")

    maleImgN, maledataN = scaleData(zzxImg, maleImg, maledata)
    femaleImgN, femaledataN = scaleData(zzxImg, femaleImg, femaledata)

    zzxdata, maledataN, femaledataN = np.array(zzxdata), np.array(maledataN), np.array(femaledataN)

    shape_diff = femaledataN - maledataN
    newShape = zzxdata + shape_diff * 0.5

    triMid = Delaunay(newShape)
    affineMatrices = findAffine(triMid, zzxdata, newShape)
    shapedImg = np.zeros(zzxImg.shape)

    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]
        affineAM = affineMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        Ymax, Xmax, layer = shapedImg.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)

        shapedImg[morphYs, morphXs] = zzxImg[imAYs, imAXs]

    maleInFemaleShape = wrapInto(maleImgN, maledataN, femaleImgN, femaledataN, 1)
    color_diff = femaleImgN - maleInFemaleShape

    affineMatrices = findAffine(triMid, femaledataN, newShape)

    producedImg = np.zeros(zzxImg.shape)

    for i in range(len(triMid.simplices)):
        tri = triMid.simplices[i]
        x1, y1 = triMid.points[tri[0]]
        x2, y2 = triMid.points[tri[1]]
        x3, y3 = triMid.points[tri[2]]
        affineAM = affineMatrices[i]

        morphedPoly = np.array(polygon(np.array([x1, x2, x3]), np.array([y1, y2, y3])))
        morphXs = morphedPoly[0, :]
        morphYs = morphedPoly[1, :]

        morphedPolyScale = np.stack([morphedPoly[0, :], morphedPoly[1, :], np.ones((morphedPoly.shape[1]))], axis=0)

        imApoly = np.dot(np.linalg.inv(affineAM), morphedPolyScale)
        imApolyInt = imApoly.astype(np.uint32)[:2, :]

        imAXs = imApolyInt[0, :]
        imAYs = imApolyInt[1, :]

        Ymax, Xmax, layer = producedImg.shape

        if not np.all(imAXs < Xmax):
            imAXs = imAXs - (imAXs >= Xmax)
        if not np.all(imAYs < Ymax):
            imAYs = imAYs - (imAYs >= Ymax)

        producedImg[morphYs, morphXs] = shapedImg[morphYs, morphXs] + color_diff[imAYs, imAXs]*50


    return producedImg

def create_gif(folder):
    import imageio
    images = []
    for i in range(FRAME_NUM):
        name = folder + "/" + str(i) + ".jpg"
        img = imageio.imread(name)
        height, width, layers = img.shape

        images.append(img)
    imageio.mimsave(folder + '/movie.gif', images)




def test3():
    # plt.imshow(io.imread("images/george.jpg"))
    im = io.imread("images/zzx.jpg")
    plt.imshow(np.ones(im.shape)*255)
    p1 = process_data("george")*1/2 + process_data("zzx")*1/2
    # for p in p1:
    #     plt.scatter(p[0], p[1])
    #     plt.draw()
    g = process_data("george")
    trimid = Delaunay(p1)
    for i in trimid.simplices:
        a = g[i[0]]
        b = g[i[1]]
        c = g[i[2]]
        plt.plot((a[0], b[0]), (a[1], b[1]), "k-")
        plt.plot((b[0], c[0]), (b[1], c[1]), "k-")
        plt.plot((a[0], c[0]), (a[1], c[1]), "k-")
    plt.show()

def reshape():
    img = io.imread("BW/7.jpg")
    img = tr.resize(img, (500, 400, 3))
    io.imsave("7.jpg", img)


def main():
    args = sys.argv[1:]

    if args[0] == "test":
        test3()

    elif len(args) == 3 and args[1] == "record":
        imname = args[0]
        name = args[2]
        img = io.imread(imname)
        # img = tr.resize(img, (500, 400, 3))
        # io.imsave(imname, img)
        # assert img.shape == (500, 400, 3)
        storePoints(img, name)

    elif len(args) == 5 and args[4] == "mean":
        imname1 = args[0]
        imname2 = args[1]
        im1dataF = process_data(args[2])
        im2dataF = process_data(args[3])
        im1 = io.imread(imname1)
        im2 = io.imread(imname2)

        im2N, im2dataFN = scaleData(im1, im2, im2dataF)
        assert im1.shape == im2N.shape
        assert np.abs(im2dataFN[-1][0] - im1dataF[-1][0]) < 0.01

        morph(im1, im2N, im1dataF, im2dataFN, args[2] + "->" + args[3], 3)

    elif len(args) == 5 and args[4] == "full":
        imname1 = args[0]
        imname2 = args[1]
        im1dataF = process_data(args[2])
        im2dataF = process_data(args[3])
        im1 = io.imread(imname1)
        im2 = io.imread(imname2)
        print(np.max(im2))

        im2, im2dataF = scaleData(im1, im2, im2dataF)
        im2*=255
        assert im1.shape == im2.shape
        assert np.abs(im2dataF[-1][0] - im2dataF[-1][0]) < 0.01

        morph(im1, im2, im1dataF, im2dataF, args[2] + "_" + args[3], FRAME_NUM)

    elif len(args) == 2 and args[1] == "video":
        create_video(args[0])

    elif len(args) == 2 and args[1] == "gif":
        create_gif(args[0])

    elif len(args) == 1 and args[0] == "bair":
        bair()

    elif len(args) == 6 and args[2] == "into":
        fromImg = io.imread(args[0])
        fromData = process_data(args[1])
        toImg = io.imread(args[3])
        toData = process_data(args[4])
        alpha = args[5]
        toImg, toData = scaleData(fromImg, toImg, toData)
        toData = np.array(toData)
        result = wrapInto(fromImg, fromData, toImg, toData, float(alpha))

        io.imsave(args[1] + "_to_" + args[4] + "alpha"+ args[5] +".jpg", result)

    elif len(args) == 2 and args[0] == "tofemale":
        if args[1] == "shape":
            result = femaleShape()
            io.imsave("female_zzx_shape.jpg", result)
        elif args[1] == "color":
            result = femaleColor()
            io.imsave("female_zzx_color.jpg", result)
        elif args[1] == "both":
            result = femaleBoth()
            io.imsave("female_zzx_both.jpg", result)

    elif len(args) == 1 and args[0] == "reshape":
        reshape()



if __name__ == "__main__":
    main()
