import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from tkinter.tix import IMAGETEXT
import cv2 as cv
from tkinter import *
from tkinter import ttk, Canvas
from tkinter import filedialog
import tkinter as tk
from custom_hovertip import CustomTooltipLabel
from sewar.full_ref import psnr, vifp
import tkinter.ttk as ttk
import PIL.Image
import PIL.ImageTk
import os
import matplotlib
matplotlib.use('TkAgg')
pro = Tk()
w = pro.winfo_screenwidth()
h = pro.winfo_screenheight()
pro.geometry("%dx%d+0+0" % (w, h))
pro.resizable(True, True)      # Non resizable width - heghit
pro.maxsize(1920, 1080)
pro.title("Eslam Mohamed - ToolBox")
pro.config(background='#99d98c')
pro.iconbitmap(r'icons\amazing.ico')
path = r"images/messi.png"  # initial test photo


################ Frames of App ##################
fr0 = Frame(width=w, height=80,
            bg='#52b69a', border=0)  # Upper frame
fr0.place(x=0, y=0)

fr1 = Frame(width='260', height=h,
            bg='#355070', border=0)  # Left frame
fr1.place(x=0, y=80)

fr3 = Frame(width='1012', height=h, border=0,
            bg='green')  # Center frame #184e77

fr3.place(x=260, y=82)

fr2 = Frame(width='260', height=h, bg='#355070', border=0)  # Right frame
fr2.place(x=w-260, y=80)
# ----------------------------------------------------
########################image config######################################
image = cv.imread(path, 0)
height, width = image.shape
canvas = Canvas(fr3, width=width, height=h, bg='#99d98c',
                bd=0, highlightthickness=0, relief='ridge')
canvas.place(x=0, y=0)  # 0,0 of frame center

# ----------------------------------------------------
###############       Button icons   ##########################

BrowseIcon = PhotoImage(file=r"icons/browsee.png")
BrightnessUp = PhotoImage(file=r"icons/bright+.png")
BrightnessDown = PhotoImage(file=r"icons/bright-.png")
skwR = PhotoImage(file=r"icons/skwR.png")
skwL = PhotoImage(file=r"icons/skwL.png")
mergIcon = PhotoImage(file=r"icons/Merge.png")
saveIcon = PhotoImage(file=r"icons/save.png")
resetIcon = PhotoImage(file=r"icons/reload.png")
eqIcon = PhotoImage(file=r"icons/eq.png")
histIcon = PhotoImage(file=r"icons/histogram.png")
whiteIcon = PhotoImage(file=r"icons/white.png")
blackIcon = PhotoImage(file=r"icons/black.png")
bitIcon = PhotoImage(file=r"icons/bit.png")
sliceIcon = PhotoImage(file=r"icons/slice.png")
logIcon = PhotoImage(file=r"icons/log.png")
grayIcon = PhotoImage(file=r"icons/gray.png")
thresholdIcon = PhotoImage(file=r"icons/threshold.png")
zoomInIcon = PhotoImage(file=r"icons/zoomIN.png")
zoomOutIcon = PhotoImage(file=r"icons/zoomOUT.png")
shapeIcon = PhotoImage(file=r"icons/shape.png")

rightIcon = PhotoImage(file=r"icons/right.png")
leftIcon = PhotoImage(file=r"icons/left.png")
upIcon = PhotoImage(file=r"icons/up.png")
downIcon = PhotoImage(file=r"icons/down.png")
rotateIcon = PhotoImage(file=r"icons/rotate.png")
negativeIcon = PhotoImage(file=r"icons/negative.png")

leftrightIcon = PhotoImage(file=r"icons/left_right.png")
updownIcon = PhotoImage(file=r"icons/up_down.png")
infoIcon = PhotoImage(file=r"icons/info.png")

zippedIcon = PhotoImage(file=r"icons/zipped.png")
cropIcon = PhotoImage(file=r"icons/crop.png")

###############        Button Functions   ##########################


cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0


def mouse_crop(event, x, y, flags, param):
    global image
    oriImage = image.copy()
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1]
                           [1], refPoint[0][0]:refPoint[1][0]]
            cv.imshow("Cropped", roi)
            image = roi


def myCrop():
    global image
    global photo
    cv.namedWindow("image")
    cv.setMouseCallback("image", mouse_crop)

    i = image.copy()
    
    if not cropping:
        cv.imshow("image", i)

    elif cropping:
        cv.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.imshow("Preview Crop", i)
        image = i
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)
    cv.waitKey(0)
    # close all open windows
    cv.destroyAllWindows()
    applyCrop()


def applyCrop():
    global image
    global photo
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def open_popup():
    # a button widget which will open a
    # new window on button click
    global image
    global photo
    global orignalImg
    top = Toplevel(fr1)
    top.geometry("240x200")
    top.title("Image Information")
    top.config(background='#99d98c')
    #Label(top, text= "Hello World!", font=('Mistral 18 bold')).place(x=150,y=80)
    row = orignalImg.shape[0]
    colom = orignalImg.shape[1]
    depth = orignalImg.dtype
    top.resizable(False, False)
    lbl1 = Label(top, text=f'Numbers of rows is {row}', font=(
        'Arial 13 bold'), justify='center', padx=5, background='#99d98c')
    lbl1.place(x=0, y=20)
    lbl2 = Label(top, text=f'Numbers of Columns is {colom}', font=(
        'Arial 13 bold'), justify='center', padx=5, background='#99d98c')
    lbl2.place(x=0, y=60)
    lbl3 = Label(top, text=f'Picture Depth is {depth}', font=(
        'Arial 13 bold'), justify='center', padx=5, background='#99d98c')
    lbl3.place(x=0, y=100)
    lbl4 = Label(top, text=f'Picture size is {colom}x{row}', font=(
        'Arial 13 bold'), justify='center', padx=5, background='#99d98c')
    lbl4.place(x=0, y=140)
    lbl5 = Label(top, text=f'PSNR = {PSNR()}', font=(
        'Arial 13 bold'), justify='center', padx=5, background='#99d98c')
    lbl5.place(x=0, y=170)


def grayScale():
    # convert array matrix to  photo
    global image
    global photo
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    # print(image.shape)


def RotateImg():
    global image
    global photo

    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def translationTop():
    global image
    global photo
    Matrix = np.float32([[1, 0, 0], [0, 1, 10]])
    image = cv.warpAffine(image, Matrix, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def translationBottom():
    global image
    global photo
    Matrix = np.float32([[1, 0, 0], [0, 1, -10]])
    image = cv.warpAffine(image, Matrix, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def translationLeft():
    global image
    global photo
    Matrix = np.float32([[1, 0, 10], [0, 1, 0]])
    image = cv.warpAffine(image, Matrix, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def translationRight():
    global image
    global photo
    Matrix = np.float32([[1, 0, -10], [0, 1, 0]])
    image = cv.warpAffine(image, Matrix, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def flipUpDown():
    global image
    global photo
    image = cv.flip(image, 0)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def flipRightLeft():
    global image
    global photo
    image = cv.flip(image, 1)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def imageNegative():
    global image
    global photo
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 255 - image[i, j]
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def skwRight():
    global image
    global photo
    point_1 = np.float32([[0, 0], [0, image.shape[0]], [
                         image.shape[1], image.shape[0]]])
    point_2 = np.float32([[10, 0], [0, image.shape[0]], [
                         image.shape[1], image.shape[0]]])
    M = cv.getAffineTransform(point_1, point_2)
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def skwLeft():
    global image
    global photo
    point_1 = np.float32([[10, 0], [0, image.shape[0]], [
                         image.shape[1], image.shape[0]]])
    point_2 = np.float32([[0, 0], [0, image.shape[0]], [
                         image.shape[1], image.shape[0]]])
    M = cv.getAffineTransform(point_1, point_2)
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def increaseAmplitude():
    global image
    global photo
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] + 10
    # Convert array image to Photo
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)  # create photo in Canvas


def decreaseAmplitude():
    global image
    global photo
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = image[i, j] - 10
    # Convert array image to Photo
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)  # create photo in Canvas


def MergeBtn():
    global photo2
    global image
    global photo

    merg = np.zeros((image.shape[0], image.shape[1]))  # array of zeros
    fln2 = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image", filetypes=(
        ("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))
    image2 = PIL.Image.open(fln2)
    image2 = np.asarray(image2)
    dim2 = (image.shape[1], image.shape[0])
    image2 = cv.resize(image2, dim2, interpolation=cv.INTER_AREA)
    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image2))
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    image2 = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            merg[row, col] = image[row, col] * 0.8 + image2[row, col] * 0.2
    image = merg
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def histApply():
    global image
    global photo
    # grayScale()
    image = cv.equalizeHist(image)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def histGraphBtn():
    global image
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

# def LogarithmicTrans():
#     global image
#     global photo
#     c = 255 / np.log(1 + np.max(image))
#     log_image = c * np.log(image + 1)
#     image = np.array(log_image, dtype=np.uint8)
#     photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)


def myLog():
    global image
    global photo
    dst = image
    dst = np.array(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dst[i, j] = np.log(image[i, j] + 1)
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
    cv.convertScaleAbs(dst, dst)
    image = dst
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def myPowerUP():
    global image
    global photo
    dst = image
    dst = np.array(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dst[i, j] = np.float_power(image[i, j], 1.5)
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
    cv.convertScaleAbs(dst, dst)
    image = dst
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def myPowerDown():
    global image
    global photo
    dst = image
    dst = np.array(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dst[i, j] = np.float_power(image[i, j], 0.5)
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
    cv.convertScaleAbs(dst, dst)
    image = dst
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def Last_Bit_Plane():
    global image
    global photo
    row, column = image.shape
    img = image
    image = np.zeros((row, column), dtype='uint8')
    for i in range(row):
        for j in range(column):
            if img[i, j] & 128:
                image[i, j] = 255
            else:
                image[i, j] = 0
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def Gray_level():
    global image
    global photo
    row, column = image.shape
    dst = image
    dst = np.array(image, dtype=np.float32)
    min_range = 130
    max_range = 200
    for i in range(row):
        for j in range(column):
            if dst[i, j] > min_range and dst[i, j] < max_range:
                image[i, j] = 255
            else:
                image[i, j] = dst[i, j]
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
################## Filters #######################


def blur3x3():
    global image
    global photo
    # fr0['bg'] = 'red'
    kernal_3x3 = np.ones((3, 3), np.float32)
    kernal_3x3 = kernal_3x3 / 9
    image = cv.filter2D(image, -1, kernal_3x3)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blur5x5():
    global image
    global photo
    kernal_5X5 = np.ones((5, 5), np.float32)
    kernal_5X5 = kernal_5X5 / 25
    image = cv.filter2D(image, -1, kernal_5X5)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blurPyramid():
    global image
    global photo
    kernal_pyramid = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [
                              2, 4, 6, 4, 2], [1, 2, 3, 2, 1]], np.float32)
    kernal_pyramid = kernal_pyramid / 81
    image = cv.filter2D(image, -1, kernal_pyramid)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blurCone():
    global image
    global photo
    Kernel_cone = np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1], [
                           0, 2, 2, 2, 0], [0, 0, 1, 0, 0]], np.float32)
    Kernel_cone = Kernel_cone / 25
    image = cv.filter2D(image, -1, Kernel_cone)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blurCircle():
    global image
    global photo
    Kernel_circle = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [
                             1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], np.float32)
    Kernel_circle = Kernel_circle / 25
    image = cv.filter2D(image, -1, Kernel_circle)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blurGaussian9x9():
    global image
    global photo
    image = cv.GaussianBlur(image, (9, 9), 0)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def blurMedian():
    global image
    global photo
    image = cv.medianBlur(image, 9)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def sobelVertical():
    global image
    global photo
    kernal_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # [1,0,-1],[2,0,-2],[1,0,-1]
    # [-1,0,1],[-2,0,2],[-1,0,1]
    image = cv.filter2D(image, -1, kernal_v)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def sobelHorizontal():
    global image
    global photo
    kernal_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # [1,2,1],[0,0,0],[-1,-2,-1]
    # [-1,-2,-1],[0,0,0],[1,2,1]
    image = cv.filter2D(image, -1, kernal_h)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def sobelDiag():
    global image
    global photo
    kernal_d = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    image = cv.filter2D(image, -1, kernal_d)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def sharpImage():
    global image
    global photo
    kernal_sh = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv.filter2D(image, -1, kernal_sh)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def sobelTotal():
    global image
    global photo
    kernal_1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernal_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernal_3 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    v = cv.filter2D(image, -1, kernal_1)
    h = cv.filter2D(image, -1, kernal_2)
    digonal = cv.filter2D(image, -1, kernal_3)
    dst_1 = cv.addWeighted(v, 1, h, 1, 0.0)
    image = cv.addWeighted(dst_1, 1, digonal, 1, 0.0)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def thresholdMethod():
    global image
    global photo
    cv.threshold(image, spnRaise.get(), 255, type=cv.THRESH_BINARY, dst=image)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def laplace_Method():
    global image
    global photo
    image = cv.Laplacian(image, cv.CV_8UC1)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def img_zoomOUT():
    global image
    global photo
    width = image.shape[1]
    height = image.shape[0]
    w = width * 0.9
    h = height * 0.9
    image = cv.resize(image, [int(w), int(h)], interpolation=cv.INTER_AREA)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def img_zoomIN():
    global image
    global photo
    width = image.shape[1]
    height = image.shape[0]
    w = width / 0.9
    h = height / 0.9
    image = cv.resize(image, [int(w), int(h)], interpolation=cv.INTER_AREA)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def edgeDetectCanny():
    global image
    global photo
    image = cv.Canny(image, 100, 205)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def edgeDetectContour():
    global image
    global photo
    ret, thresh = cv.threshold(image, 100, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


def thrSh_plt():
    global image
    ret, thresh1 = cv.threshold(image, 25, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(image, 50, 255, cv.THRESH_BINARY)
    ret, thresh3 = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
    ret, thresh4 = cv.threshold(image, 150, 255, cv.THRESH_BINARY)
    ret, thresh5 = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
    titles = ['Original', '25', '50', '100', '150', '200']
    images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
    spn1['state'] = 'readonly'


def Recognation():
    # using a findContours() function
    global image
    global photo
    ret, thresh3 = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
    image = thresh3
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    i = 0
    # list for storing names of shapes
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv.approxPolyDP(
            contour, 0.01 * cv.arcLength(contour, True), True)

        # using drawContours() function
        cv.drawContours(image, [contour], 0, (0, 0, 255), 5)

        # finding center point of shape
        M = cv.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        X = x-12
        Y = y-1
        # putting shape name at center of each shape
        if len(approx) == 3:
            cv.putText(image, 'Triangle', (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        elif len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(approx)
            aspectratio = float(w)/h
            if aspectratio >= 0.95 and aspectratio <= 1.05:

                cv.putText(image, 'Square', (x, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                cv.putText(image, 'Rectangle', (x, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        elif len(approx) == 5:
            cv.putText(image, 'Pentagon', (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        elif len(approx) == 6:
            cv.putText(image, 'Hexagon', (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        elif len(approx) == 10:
            cv.putText(image, 'Star', (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        else:
            cv.putText(image, 'circle', (X, Y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

# def thresholdBtn():
#     global finalEdit
#     global image
#     global photo
#     ret1, thresh1 = cv.threshold(image, spnRaise.get(), 255, cv.THRESH_BINARY_INV)
#     image = thresh1
#     photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)

#################filter Setting Button#############


def selected(event):
    if clicked.get() == 'Blur 3x3':
        blur3x3()
    elif clicked.get() == 'Blur 5x5':
        blur5x5()
    elif clicked.get() == 'Pyramid':
        blurPyramid()
    elif clicked.get() == 'Cone':
        blurCone()
    elif clicked.get() == 'Circular':
        blurCircle()
    elif clicked.get() == 'Gaussian 9x9':
        blurGaussian9x9()
    elif clicked.get() == 'Median':
        blurMedian()
    elif clicked.get() == 'Low-Pass (select Value)':
        Low_pass_filter()


blurOptions = ['Blur 3x3', 'Blur 5x5', 'Pyramid',
               'Cone', 'Circular', 'Gaussian 9x9', 'Median', 'Low-Pass (select Value)']

clicked = StringVar()
clicked.set('Blur Filters')


def selected2(event):
    if clicked2.get() == 'Sobel-Horizonatl':
        sobelHorizontal()
    elif clicked2.get() == 'Sobel-Vertical':
        sobelVertical()
    elif clicked2.get() == 'Sobel-Diagonal':
        sobelDiag()
    elif clicked2.get() == 'Sobel':
        sobelTotal()
    elif clicked2.get() == 'Threshold':
        thresholdMethod()
    elif clicked2.get() == 'LaPlace':
        laplace_Method()
    elif clicked2.get() == 'Canny':
        edgeDetectCanny()
    elif clicked2.get() == 'Contour':
        edgeDetectContour()
    elif clicked2.get() == 'Sharp':
        sharpImage()


edgeOptions = ['Sobel', 'Sobel-Horizonatl',
               'Sobel-Vertical', 'Sobel-Diagonal', 'LaPlace', 'Canny', 'Contour', 'Threshold', 'Sharp']

clicked2 = StringVar()
clicked2.set('Edge Detect')

spnRaise = IntVar()

###################################################


def Browse():
    global image
    global photo
    global orignalImg
    global finalEdit
    global imgcrp
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open image", filetypes=(
        ("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))  # image extensions
    image = PIL.Image.open(fln)
    image = np.asarray(image)  # convert img2array
    orignalImg = image  # save orignal
    imgcrp = image  # save orignal
    finalEdit = image
    if image.shape[0] > 700:  # height
        image = cv.resize(image, [image.shape[1], 700])
    if image.shape[1] > 1012:  # weidth
        image = cv.resize(image, [1012, image.shape[0]])

    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)

    #bt1['state'] = NORMAL
    bt1.state = NORMAL
    bt2.state = NORMAL
    bt3.state = NORMAL
    bt4.state = NORMAL
    bt5.state = NORMAL
    bt6.state = NORMAL
    bt7.state = NORMAL
    bt8.state = NORMAL
    bt9.state = NORMAL
    bt10.state = NORMAL
    btn12.state = NORMAL
    btn13.state = NORMAL
    btn14.state = NORMAL
    btn15.state = NORMAL
    btn16.state = NORMAL
    btn17.state = NORMAL
    btn18.state = NORMAL
    btn19.state = NORMAL
    btn20.state = NORMAL
    btn21.state = NORMAL
    btn22.state = NORMAL
    btn23.state = NORMAL
    btn24.state = NORMAL
    btn25.state = NORMAL
    mnuBar.state = NORMAL
    mnuBar2.state = NORMAL
    # spn1['state']='readonly'
    btn26.state = NORMAL
    btn27.state = NORMAL
    btn28.state = NORMAL
    btn29.state = NORMAL
    spn2['state'] = NORMAL
    mnuBar3.state = NORMAL
    btn30.state = NORMAL
    btn31.state = NORMAL


def SaveAs():
    global image
    global photo
    global orignalImg
    global finalEdit
    fS = filedialog.asksaveasfilename(filetypes=(
        ("JPG File", "*.jpg"), ("PNG File", "*.png")), defaultextension=".jpg", title="Save As")
    cv.imwrite(fS, image)


def SaveAsCompress():
    global image
    global photo
    global orignalImg
    global finalEdit
    fS = filedialog.asksaveasfilename(filetypes=(
        ("JPG File", "*.jpg"), ("PNG File", "*.png")), defaultextension=".jpg", title="Save As")
    cv.imwrite(fS, image, [int(cv.IMWRITE_JPEG_QUALITY), 50])


def resets():
    global image
    global orignalImg
    global finalEdit
    global photo
    image = orignalImg
    finalEdit = orignalImg
    if image.shape[0] > 700:  # height
        image = cv.resize(image, [image.shape[1], 700])
    if image.shape[1] > 1012:  # weidth
        image = cv.resize(image, [1012, image.shape[0]])
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)


spnRaise2 = IntVar()


def Low_pass_filter():  # image in frequency domain in low freq
    global image
    global photo

    image = np.fft.fft2(image)
    shiftimage = np.fft.fftshift(image)
    # Filter: Low pass filter
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = spnRaise2.get()
    #D0 = 80
    n = 3
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    # Ideal Low Pass Filtering
    gshift = shiftimage * H
    G = np.fft.ifftshift(gshift)
    image = np.abs(np.fft.ifft2(G))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
##############################################################

###################### Themes ################################


def thm1():
    # 457b9d upper fr0
    # 1d3557 left and right fr1 fr2
    # f1faee center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#457b9d')
    fr1.config(bg='#1d3557')
    fr2.config(bg='#1d3557')
    fr3.config(bg='#1d3557')
    canvas.config(bg='#f1faee')
    pro.config(background='#f1faee')


def thm2():
    # FB2576 upper fr0
    # 3F0071 left and right fr1 fr2
    # E5B8F4 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#FB2576')
    fr1.config(bg='#3F0071')
    fr2.config(bg='#3F0071')
    fr3.config(bg='#3F0071')
    canvas.config(bg='#E5B8F4')
    pro.config(background='#E5B8F4')


def thm3():
    # da2c38 upper fr0
    # 226f54 left and right fr1 fr2
    # 87c38f center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#da2c38')
    fr1.config(bg='#226f54')
    fr2.config(bg='#226f54')
    fr3.config(bg='#226f54')
    canvas.config(bg='#87c38f')
    pro.config(background='#87c38f')


def thm4():
    # 432818 upper fr0
    # 99582a left and right fr1 fr2
    # bb9457 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#432818')
    fr1.config(bg='#99582a')
    fr2.config(bg='#99582a')
    fr3.config(bg='#99582a')
    canvas.config(bg='#bb9457')
    pro.config(background='#bb9457')


def thm5():
    # 82c0cc upper fr0
    # ffa62b left and right fr1 fr2
    # ede7e3 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#db5461')
    fr1.config(bg='#37505c')
    fr2.config(bg='#37505c')
    fr3.config(bg='#37505c')
    canvas.config(bg='#8ca0d7')
    pro.config(background='#8ca0d7')


def thm6():
    # 0a2b4e upper fr0
    # a71814 left and right fr1 fr2
    # #d9e2e7 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#0a2b4e')
    fr1.config(bg='#a71814')
    fr2.config(bg='#a71814')
    fr3.config(bg='#a71814')
    canvas.config(bg='#447BBE')
    pro.config(background='#447BBE')


def thm7():
    # c0c0c0 upper fr0
    # ff6700 left and right fr1 fr2
    # ebebeb center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#d65108')
    fr1.config(bg='#023047')
    fr2.config(bg='#023047')
    fr3.config(bg='#023047')
    canvas.config(bg='#fbb13c')
    pro.config(background='#fbb13c')


def thm8():
    # e0aaff upper fr0
    # #441d66 left and right fr1 fr2
    # 875399 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='#c05299')
    fr1.config(bg='#441d66')
    fr2.config(bg='#441d66')
    fr3.config(bg='#441d66')
    canvas.config(bg='#875399')
    pro.config(background='#875399')


def thm9():
    # e0aaff upper fr0
    # #441d66 left and right fr1 fr2
    # 875399 center  canvas
    global fr0
    global fr1
    global fr2
    fr0.config(bg='gray19')
    fr1.config(bg='black')
    fr2.config(bg='black')
    fr3.config(bg='black')
    canvas.config(bg='gray10')
    pro.config(background='gray10')


#############################################
themeOptions = ['Theme 1', 'Theme 2',
                'Theme 3', 'Theme 4', 'Theme 5', 'Theme 6', 'Theme 7', 'Theme 8', 'Theme 9']

clicked3 = StringVar()
clicked3.set('App Themes')


def selected3(event):
    if clicked3.get() == 'Theme 1':
        thm1()
    elif clicked3.get() == 'Theme 2':
        thm2()
    elif clicked3.get() == 'Theme 3':
        thm3()
    elif clicked3.get() == 'Theme 4':
        thm4()
    elif clicked3.get() == 'Theme 5':
        thm5()
    elif clicked3.get() == 'Theme 6':
        thm6()
    elif clicked3.get() == 'Theme 7':
        thm7()
    elif clicked3.get() == 'Theme 8':
        thm8()
    elif clicked3.get() == 'Theme 9':
        thm9()


def PSNR():
    global image
    global orignalImg
    OG = orignalImg
    OG = cv.cvtColor(OG, cv.COLOR_BGR2GRAY)
    SNR = str(psnr(OG, image))
    return SNR


############################ Buttons ##########################
bt1 = ctk.CTkButton(fr1, text='Gray Scale', command=grayScale, text_color='black',
                    corner_radius=15, image=grayIcon, width=8, hover_color='springgreen', fg_color='snow3')
bt1.place(x=60, y=20)

bt1.state = DISABLED
#hov1=Hovertip(bt1,'Convert from RGB to Gray',hover_delay=1000)
CustomTooltipLabel(bt1, 'Convert from RGB to Gray', border=1,
                   hover_delay=500, background='springgreen')

bt2 = ctk.CTkButton(fr2, text='', command=translationRight, text_color='black',
                    corner_radius=10, image=leftIcon, width=10, hover_color='springgreen', fg_color='#E3CF57')
bt2.place(x=90, y=30)
bt2.state = DISABLED
CustomTooltipLabel(bt2, 'Move the photo left with crop', border=1,
                   hover_delay=500, background='springgreen', wraplength=50)

bt3 = ctk.CTkButton(fr2, text='', command=translationLeft, text_color='black',
                    corner_radius=10, image=rightIcon, width=10, hover_color='springgreen', fg_color='#E3CF57')
bt3.place(x=140, y=30)
bt3.state = DISABLED
CustomTooltipLabel(bt3, 'Move the photo right with crop', border=1,
                   hover_delay=500, background='springgreen', wraplength=50)

bt4 = ctk.CTkButton(fr2, text='', command=translationTop, text_color='black',
                    corner_radius=10, image=downIcon, width=10, hover_color='springgreen', fg_color='#E3CF57')
bt4.place(x=180, y=30)
bt4.state = DISABLED
CustomTooltipLabel(bt4, 'Move Down', border=1,
                   hover_delay=500, background='springgreen', wraplength=40)

bt5 = ctk.CTkButton(fr2, text='', command=translationBottom, text_color='black',
                    corner_radius=10, image=upIcon, width=10, hover_color='springgreen', fg_color='#E3CF57')
bt5.place(x=60, y=30)
bt5.state = DISABLED
CustomTooltipLabel(bt5, 'Move Up', border=1,
                   hover_delay=500, background='springgreen', wraplength=40)

bt6 = ctk.CTkButton(fr2, text='', command=RotateImg, text_color='black',
                    corner_radius=10, image=rotateIcon, width=10, hover_color='springgreen', fg_color='lightblue1')
bt6.place(x=60, y=80)
bt6.state = DISABLED
CustomTooltipLabel(bt6, 'Rotate the photo arrount it self', border=1,
                   hover_delay=500, background='springgreen')

bt7 = ctk.CTkButton(fr2, text='Negative', command=imageNegative, text_color='black',
                    corner_radius=4, image=negativeIcon, width=8, hover_color='springgreen', fg_color='lightblue1')
bt7.place(x=100, y=80)
bt7.state = DISABLED
CustomTooltipLabel(bt7, 'Negative all pixels colors in photo', border=1,
                   hover_delay=500, background='springgreen', wraplength=50)

bt8 = ctk.CTkButton(fr2, text='', command=flipRightLeft, text_color='black',
                    corner_radius=4, image=leftrightIcon, width=8, hover_color='springgreen', fg_color='#FF7D40')
bt8.place(x=60, y=130)
bt8.state = DISABLED
CustomTooltipLabel(bt8, 'Flip the image Right and Left', border=1,
                   hover_delay=500, background='springgreen')

bt9 = ctk.CTkButton(fr2, text='', command=flipUpDown, text_color='black',
                    corner_radius=4, image=updownIcon, width=8, hover_color='springgreen', fg_color='#FF7D40')
bt9.place(x=165, y=130)
bt9.state = DISABLED
CustomTooltipLabel(bt9, 'Flip the image Up and Down', border=1,
                   hover_delay=1000, background='springgreen', wraplength=60)

bt10 = ctk.CTkButton(master=fr0, text="Reset", fg_color='orange',
                     command=resets, text_color='black', image=resetIcon, width=10, hover_color='red2')
bt10.place(x=150, y=15)
bt10.state = DISABLED
CustomTooltipLabel(bt10, 'Recover Orignal Image', border=1,
                   hover_delay=1000, background='orange')

btn11 = ctk.CTkButton(master=fr0, text="Browse", fg_color='aqua',
                      command=Browse, text_color='black', image=BrowseIcon, width=10, hover_color='khaki1')
btn11.place(x=30, y=15)
# myTip11 = Hovertip(btn11,'Upload Photo from your local files',hover_delay=100)
CustomTooltipLabel(btn11, 'Upload Photo from your local files',
                   border=1, hover_delay=1000, background='aqua')

btn12 = ctk.CTkButton(master=fr1, text="Skew Right", fg_color='blanchedalmond',
                      command=skwRight, text_color='black', image=skwR, width=10, corner_radius=17, hover_color='springgreen')
btn12.place(x=60, y=140)
btn12.state = DISABLED
CustomTooltipLabel(btn12, 'change direction of image to right',
                   border=1, hover_delay=1000, background='blanchedalmond')

btn13 = ctk.CTkButton(master=fr1, text="Skew Left", fg_color='blanchedalmond',
                      command=skwLeft, text_color='black', image=skwL, width=10, corner_radius=17, hover_color='springgreen')
btn13.place(x=60, y=200)
btn13.state = DISABLED
CustomTooltipLabel(btn13, 'change direction of image to left',
                   border=1, hover_delay=1000, background='blanchedalmond')

btn14 = ctk.CTkButton(master=fr0, text="Brightness +", fg_color='blanchedalmond',
                      command=increaseAmplitude, text_color='black', corner_radius=10, image=BrightnessUp, width=10, hover_color='springgreen')
btn14.state = DISABLED
#btn14.place(x=590, y=10)

btn15 = ctk.CTkButton(master=fr0, text="Brightness -", fg_color='blanchedalmond',
                      command=decreaseAmplitude, text_color='black', corner_radius=10, image=BrightnessDown, width=10, hover_color='springgreen')
btn15.state = DISABLED
#btn15.place(x=790, y=10)

btn16 = ctk.CTkButton(master=fr1, text="Merge", fg_color='#f19c79',
                      command=MergeBtn, text_color='black', corner_radius=15, image=mergIcon, width=10, hover_color='springgreen')
btn16.state = DISABLED
btn16.place(x=65, y=270)
CustomTooltipLabel(btn16, 'Use Blind to add two images in one view',
                   border=1, hover_delay=1000, background='blanchedalmond')

btn17 = ctk.CTkButton(master=fr0, text="", fg_color='orange',
                      command=open_popup, text_color='black', corner_radius=10, image=infoIcon, width=10, hover_color='springgreen')
btn17.state = DISABLED
btn17.place(x=1330, y=15)
CustomTooltipLabel(btn17, 'Image information', border=1,
                   hover_delay=1000, background='orange')

btn18 = ctk.CTkButton(master=fr2, text="Hist-Equalize", fg_color='blanchedalmond',
                      command=histApply, text_color='black', corner_radius=20, image=eqIcon, width=10, hover_color='springgreen')
btn18.state = DISABLED
btn18.place(x=65, y=540)
CustomTooltipLabel(btn18, 'Equalize pixels among X axis', border=1,
                   hover_delay=1000, background='springgreen')

btn19 = ctk.CTkButton(master=fr2, text="Hist-Graph", fg_color='blanchedalmond',
                      command=histGraphBtn, text_color='black', corner_radius=20, image=histIcon, width=10, hover_color='springgreen')
btn19.state = DISABLED
btn19.place(x=65, y=490)
CustomTooltipLabel(btn19, 'Graph shows pixels value on it', border=1,
                   hover_delay=1000, background='springgreen')

btn20 = ctk.CTkButton(master=fr2, text="Logaritmic", fg_color='blanchedalmond',
                      command=myLog, text_color='black', corner_radius=15, image=logIcon, width=10, hover_color='springgreen')

btn20.place(x=65, y=590)
btn20.state = DISABLED
CustomTooltipLabel(btn20, 'map a narrow range of dark input values into a wider range', border=1,
                   hover_delay=1000, background='springgreen', wraplength=100)

btn21 = ctk.CTkButton(master=fr1, text="Power 0.5", fg_color='white',
                      command=myPowerDown, text_color='black', corner_radius=15, image=whiteIcon, width=8, hover_color='springgreen')

btn21.place(x=65, y=640)
btn21.state = DISABLED
CustomTooltipLabel(btn21, 'map a narrow range of dark to whiten by 0.5', border=1,
                   hover_delay=1000, background='springgreen')

btn22 = ctk.CTkButton(master=fr1, text="Power 1.5",  fg_color='white',
                      command=myPowerUP, text_color='black', corner_radius=15, image=blackIcon, width=8, hover_color='springgreen')

btn22.place(x=65, y=590)
btn22.state = DISABLED
CustomTooltipLabel(btn22, 'map a narrow range of whiten to dark by 1.5', border=1,
                   hover_delay=1000, background='springgreen')

btn23 = ctk.CTkButton(master=fr1, text="Last BitPlan",  fg_color='blanchedalmond',
                      command=Last_Bit_Plane, text_color='black', corner_radius=15, image=bitIcon, width=8, hover_color='springgreen')

btn23.place(x=65, y=530)
btn23.state = DISABLED
CustomTooltipLabel(btn23, 'Converting a gray level image to a binary image', border=1,
                   hover_delay=1000, background='springgreen', wraplength=60)

btn24 = ctk.CTkButton(master=fr1, text="Slicing",  fg_color='blanchedalmond',
                      command=Gray_level, text_color='black', corner_radius=15, image=sliceIcon, width=8, hover_color='springgreen')

btn24.place(x=65, y=460)
btn24.state = DISABLED
CustomTooltipLabel(btn24, 'way to highlight gray range', border=1,
                   hover_delay=1000, background='springgreen', wraplength=60)


mnuBar = ctk.CTkOptionMenu(master=fr2, variable=clicked, values=blurOptions, command=selected, button_color='orange', fg_color='#99d98c',
                           button_hover_color='gold', text_color='black', dropdown_hover_color='gold', dropdown_text_color='black', dropdown_color='white', corner_radius=25)
mnuBar.place(x=65, y=295)
mnuBar.state = DISABLED

mnuBar2 = ctk.CTkOptionMenu(master=fr2, variable=clicked2, values=edgeOptions, command=selected2, button_color='orange', fg_color='#99d98c',
                            button_hover_color='gold', text_color='black', dropdown_hover_color='gold', dropdown_text_color='black', dropdown_color='white', corner_radius=25)
mnuBar2.place(x=65, y=335)
mnuBar2.state = DISABLED


spn1 = Spinbox(master=fr2, bg='green', from_=0, to=255, width=5, fg='black',
               readonlybackground='#99d98c', increment=5, textvariable=spnRaise, justify='center', relief='flat')
spn1.place(x=55, y=385)
spn1['state'] = DISABLED

lbl1 = ctk.CTkLabel(master=fr2, text='Threshold Value',
                    text_font=("Arial", "10"))
lbl1.place(x=100, y=380)

spn2 = Spinbox(master=fr2, bg='#90e0ef', values=[50, 80, 100, 150, 200], width=5, fg='black',
               readonlybackground='#90e0ef', increment=5, textvariable=spnRaise2, justify='center', relief='flat')
spn2.place(x=55, y=249)
spn2['state'] = DISABLED


lbl2 = ctk.CTkLabel(master=fr2, text='Low Pass Value',
                    text_font=("Arial", "10"))
lbl2.place(x=100, y=245)


btn25 = ctk.CTkButton(master=fr2, text="Threshold View",  fg_color='blanchedalmond',
                      command=thrSh_plt, text_color='black', corner_radius=15, image=thresholdIcon, width=8, hover_color='springgreen')

btn25.place(x=65, y=435)
btn25.state = DISABLED
CustomTooltipLabel(btn25, 'See your best view of threshold before apply', border=1,
                   hover_delay=1000, background='springgreen', wraplength=90)

btn26 = ctk.CTkButton(master=fr0, text="Zoom In",  fg_color='#72efdd',
                      command=img_zoomIN, text_color='black',
                      corner_radius=15, image=zoomInIcon, width=8, hover_color='#e0aaff')

btn26.place(x=390, y=15)
btn26.state = DISABLED

btn27 = ctk.CTkButton(master=fr0, text="Zoom Out",  fg_color='#72efdd',
                      command=img_zoomOUT, text_color='black',
                      corner_radius=15, image=zoomOutIcon, width=8, hover_color='#e0aaff')

btn27.place(x=1000, y=15)
btn27.state = DISABLED


btn28 = ctk.CTkButton(master=fr2, text="Shape Detect", fg_color='mediumorchid1',
                      command=Recognation, text_color='black', corner_radius=10, image=shapeIcon, width=10, hover_color='springgreen')
btn28.state = DISABLED
btn28.place(x=65, y=650)
CustomTooltipLabel(btn28, 'Detect Geometric Shapes', border=1,
                   hover_delay=1000, background='springgreen')

btn29 = ctk.CTkButton(master=fr0, text="Save As", fg_color='aqua',
                      command=SaveAs, text_color='black', corner_radius=10, image=saveIcon, width=10, hover_color='khaki1')

btn29.place(x=1380, y=15)
btn29.state = DISABLED

mnuBar3 = ctk.CTkOptionMenu(master=fr0, variable=clicked3, values=themeOptions, command=selected3, button_color='orange', fg_color='#99d98c',
                            button_hover_color='gold', text_color='black', dropdown_hover_color='gold', dropdown_text_color='black', dropdown_color='white', corner_radius=25)
mnuBar3.place(x=680, y=15)
mnuBar3.state = DISABLED

btn30 = ctk.CTkButton(master=fr0, text="", fg_color='mediumorchid1',
                      command=SaveAsCompress, text_color='black', corner_radius=10, image=zippedIcon, width=10, hover_color='springgreen')
btn30.state = DISABLED
btn30.place(x=1285, y=15)
CustomTooltipLabel(btn30, 'Save With Compression in JPG', border=1,
                   hover_delay=1000, background='mediumorchid1')

btn31 = ctk.CTkButton(master=fr1, text="Crop", fg_color='indianred1',
                      command=myCrop, text_color='black', corner_radius=15, image=cropIcon, width=10, hover_color='springgreen')
btn31.place(x=65, y=360)
btn31.state = DISABLED
CustomTooltipLabel(btn31, 'Select best part of image and crop it',
                   border=1, hover_delay=1000, background='mediumorchid1')

pro.mainloop()
