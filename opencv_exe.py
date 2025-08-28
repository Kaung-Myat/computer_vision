import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import  sys
import  os

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

    @staticmethod
    def image_threshold():
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_filter = cv.medianBlur(gray,5)
        th1 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

        titles = ['Original','Median Filter','Mean Thresholding','Gaussian Thresholding']
        images = [gray,image_filter,th1,th2]
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.imshow(images[i],'gray')

            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


class Video:
    @staticmethod
    def video_capture():
        cam = cv.VideoCapture(0, cv.CAP_DSHOW)

        if not cam.isOpened():
            print("Error: Cannot open camera")
            return

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Error: Cannot retrieve frame")
                break
            video_color_reduce = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            cv.imshow("Video Capture", video_color_reduce)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv.destroyAllWindows()

    @staticmethod
    def img_to_video():
        # Step 1: Load all PNG images from the folder (sorted by filename order)
        images = sorted(glob.glob("assets/video_capture_images/*.png"))

        if not images:
            print("Error: No images found in folder.")
            return

        # Step 2: Read the first image to get width & height
        frame = cv.imread(images[0])
        height, width, _ = frame.shape

        # Step 3: Define the video writer (filename, codec, fps, frame_size)
        out = cv.VideoWriter(
            "output.avi",
            cv.VideoWriter_fourcc(*'XVID'),  # Corrected typo here
            24,  # Frames per second
            (width, height)  # Frame size
        )

        # Step 4: How many times to repeat each image (for slower playback)
        repeat_count = 10

        # Step 5: Loop over all images
        for img_path in images:
            img = cv.imread(img_path)  # Corrected from "img - ..."
            if img is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue

            # Repeat each frame multiple times to control duration
            for _ in range(repeat_count):
                out.write(img)

        # Step 6: Release video writer
        out.release()
        print("Video saved as output.avi")


class Detection:
    @staticmethod
    def image_face_detection():
        # The name of the folder containing the images
        IMAGE_FOLDER = 'assets/images'
        try:
            image_filename = sys.argv[1]
        except IndexError:
            image_filename = 'person.jpg'  # Default image

        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        # Paths to all Haar Cascade XML files
        face_cascade_path = 'haarcascade_frontalface_default.xml'
        eye_cascade_path = 'haarcascade_eye.xml'
        smile_cascade_path = 'haarcascade_smile.xml'

        # --- 2. Load All Classifiers ---
        face_cascade = cv.CascadeClassifier(face_cascade_path)
        eye_cascade = cv.CascadeClassifier(eye_cascade_path)
        smile_cascade = cv.CascadeClassifier(smile_cascade_path)

        # --- 3. Load Image and Convert to Grayscale ---
        image = cv.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image from path: {image_path}")
            sys.exit()

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # --- 4. Detect Faces ---
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        print(f"✅ Found {len(faces)} faces in '{image_filename}'!")

        # --- 5. Loop Through Faces and Detect Features within Each Face ---
        # Loop over each detected face (x, y, width, height)
        for (x, y, w, h) in faces:
            # Draw a green rectangle around the face
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create a Region of Interest (ROI) for the face in both color and grayscale
            # This means we will only search for eyes and mouths inside this face region
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            # Draw blue rectangles around the eyes
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # Detect mouth (smile) within the face ROI
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,  # Scale factor is often higher for smiles
                minNeighbors=20,
                minSize=(25, 25)
            )
            # Draw a red rectangle around the mouth
            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        # --- 6. Display the final image ---

        cv.imshow("Face and Feature Detection", image)
        cv.waitKey(0)
        cv.destroyAllWindows()