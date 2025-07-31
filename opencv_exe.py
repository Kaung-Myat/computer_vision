import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("assets/robot.jpeg")
height, width = img.shape[:2]


class Image:
    @staticmethod
    def read_image():
        if img is None:
            print("Failed to load image.")
            return
        cv.imshow("Robot", img)

    @staticmethod
    def write_image():
        if img is None:
            print("Failed to load image.")
            return
        saved = cv.imwrite("assets/opencv_logo.png", img)
        if saved:
            print("Image saved successfully as 'assets/opencv_logo.png'")
        else:
            print("Failed to save image.")
        cv.imshow("Saved Image", img)

class Shape:
    @staticmethod
    def shape():
        cv.line(img,(20,400),(400,20),(255,255,255),3)
        cv.rectangle(img,(200,100),(400,400),(0,255,0),5)
        cv.ellipse(img,(300,425),(80,20),5,0,360,(0,0,255),-1)
        cv.imshow("Robot with line", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


    @staticmethod
    def put_text():
        txt = "Hello"
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img,txt,(10,100),font,2,(255,255,255),2,cv.LINE_AA)
        cv.putText(img,"Hi",(10,400),font,2,(255,0,0),2,cv.LINE_4)
        cv.imshow("Image with text",img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def resize_img():
        small_img = cv.resize(img,(int(width/5),int(height/5)),interpolation=cv.INTER_AREA)
        large_img= cv.resize(img,(int(width*2),int(height*2)),interpolation=cv.INTER_LANCZOS4)
        cv.imshow("Image Resize",small_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def rotate_img():
        # Define the center of rotation (image center)
        center = (width // 2, height // 2)

        # Define angle and scale
        angle = 180  # degrees
        scale = 1.0  # no scaling

        # Get the rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)

        # Apply the rotation using warpAffine
        rotated_img = cv.warpAffine(img, rotation_matrix, (width, height))
        cv.imshow("Rotated Image",rotated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

class Color:
    @staticmethod
    def color_transform():
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lab_image = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        ycrcb_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        # cv.imshow('Original', img)
        # cv.imshow('HSV', hsv_image)
        # cv.imshow('LAB', lab_image)
        # cv.imshow('YCRCB', ycrcb_image)

        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(img,kernel,iterations=1)
        dilation = cv.dilate(img,kernel,iterations=1)

        cv.imshow('Erosion',erosion)
        cv.imshow('Dilation',dilation)

    @staticmethod
    def histogram():
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist,color=col)
            plt.xlim([0,256])
        plt.show()

    @staticmethod
    def gaussian_blur():
        gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and avoid false edge detection
        blurred_image = cv.GaussianBlur(gray_image,(5,5),0)

        #Apply canny edge detection
        edges = cv.Canny(blurred_image,threshold1=100,threshold2=200)

        #show the original and the edge-detected images
        plt.subplot(1,2,1)
        plt.title('Original Image')
        plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        plt.axis('off') # Hide x and y axis

        plt.subplot(1,2,2)
        plt.title('Edge Detection')
        plt.imshow(edges,cmap='gray')
        plt.axis('off')

        plt.show()
