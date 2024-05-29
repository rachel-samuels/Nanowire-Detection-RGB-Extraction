from tkinter import Tk, Canvas, Scrollbar, Button, Frame, BOTH, HORIZONTAL, VERTICAL, NW, filedialog, Entry, Label
import csv
import os
from tkinter import messagebox
from tkinter import IntVar, Checkbutton
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from MTM import matchTemplates, drawBoxesOnRGB
import math
import scipy.io
import ast
import pandas as pd
import tkinter as tk
from tkinter import simpledialog
import shutil
import time
from scipy.stats import pearsonr


#1. Directly Fit an Ellipse with Additional Checks
# Instead of relying solely on the rectangle fitting, you could directly fit an ellipse to the contour using cv2.fitEllipse and then apply additional checks:
# Check the ratio of the axes of the ellipse and compare it with your thresholds.
# Implement a check for the thickness of the object by calculating the area of the fitted ellipse and comparing it with the contour area. A significant difference might indicate an overshoot.
# 2. Custom Aspect Ratio Calculation for Thin Objects
# For very thin objects, the aspect ratio calculated from the width and height of the bounding rectangle might not be reliable. You could calculate the aspect ratio using the moments of the contour to get a more accurate measure of the object's elongation:
# 3. Use Convex Hull to Avoid Small Artifacts
# Before fitting an ellipse or rectangle, consider using the convex hull of the contour. This can help smooth out small irregularities or artifacts that might cause an overshoot in the ellipse fitting:
# 4. Dynamic Thresholding Based on Object Size
# Adjust your criteria dynamically based on the size of the object. For instance, very thin or small objects might require stricter thresholds for aspect ratios or axis lengths to prevent overshoots. You could scale your major_axis_min and minor_axis_max based on the detected area of the object:


def enhance_contrast(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))  # Adjust parameters here
    l_clahe = clahe.apply(l)
    
    # Merge the updated L channel with the original A and B channels
    updated_lab_image = cv2.merge((l_clahe, a, b))
    
    # Convert LAB back to BGR
    result_image = cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)
    
    return result_image

def filter_contours_by_criteria(contours, min_pixels, max_pixels, aspect_ratio_min, aspect_ratio_max, major_axis_min, minor_axis_max, reference_size=100):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_pixels <= area <= max_pixels):
            continue

        # Calculate moments for eccentricity calculation
        moments = cv2.moments(contour)
        if moments['mu20'] + moments['mu02'] > 0:  # Avoid division by zero
            eccentricity = np.sqrt(1 - (2 * moments['mu11']**2) / ((moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2)) if (moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2 > 0 else 0

            # Use convex hull to smooth contour
            hull = cv2.convexHull(contour)

            # Only proceed if contour has more than 4 points for ellipse fitting
            if len(hull) >= 5:
                ellipse = cv2.fitEllipse(hull)
                (x, y), (minor_axis, major_axis), angle = ellipse

                major_axis -= minor_axis
                major_axis /= 2
                # Calculate the aspect ratio of the fitted ellipse
                Aspect_ratio =  minor_axis/ major_axis if major_axis > 0 else 0

                # Apply modified criteria including eccentricity for shape refinement
                if aspect_ratio_min < Aspect_ratio <= aspect_ratio_max and major_axis > major_axis_min and minor_axis < minor_axis_max and eccentricity < 1:
                    filtered_contours.append(contour)

    return filtered_contours



class ImageScrollApp:
    def __init__(self, root):
        self.root = root
        self.invert_image_var = IntVar()  # To track the state of the checkbox
        self.invert_binary_var = IntVar()  # New IntVar for inverting the binary image
        self.setup_ui()
        self.binary_image = None
        self.rotated_rectangles = []
        self.fill_values = {}
        self.image = None
        self.canvas_width=None
        self.canvas_height=None
        self.last_button_pressed = None  
        
        self.click_count = 0
        self.background_color = None
        self.dot_colors = []
        self.center_color = None
        self.params ={}
        self.params['ImageThreshold'] = 0  # Default value
        self.params['SubtractBackground'] = 1
        self.params['AnnotationColor'] = (0, 0, 255)
        self.params['CheckSum'] = 0
        self.selected_region_gray = None
        self.new_image_path = None
        self.tag = []

    def setup_ui(self):
        # Frame for buttons
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side="top", fill="x")

        # Frame for parameters
        self.param_frame = Frame(self.root)
        self.param_frame.pack(side="left", fill="y")

        # Add a checkbox for inverting the image
        self.invert_image_checkbox = Checkbutton(self.button_frame, text="Invert Image", variable=self.invert_image_var, onvalue=1, offvalue=0)
        self.invert_image_checkbox.pack(side="left")

        # Load and process image for lithotag detection
        self.lithotag_detection_button = Button(self.button_frame, text="Lithotag Detection",
                                                command=self.prepare_for_lithotag_detection)
        self.lithotag_detection_button.pack(side="left")

        # Button for showing binary image
        self.show_binary_image_button = Button(self.button_frame, text="Show Binary Image",
                                               command=self.show_binary_image)
        self.show_binary_image_button.pack(side="left")

        self.invert_binary_var.set(0)

        self.invert_binary_checkbox = Checkbutton(self.button_frame, text="Invert Binary Image",
                                                  variable=self.invert_binary_var, onvalue=1, offvalue=0)
        self.invert_binary_checkbox.pack()
        self.invert_binary_checkbox.pack(side="left")

        self.find_contours_button = Button(self.button_frame, text="Find Contours", command=self.prepare_for_contour_removal)
        self.find_contours_button.pack(side="left")

        self.plot_rgb_button = Button(self.button_frame, text="Plot RGB Colors",
                                      command=self.plot_and_save_contours_rgb)
        self.plot_rgb_button.pack(side="left")

 

        # Parameters
        Label(self.param_frame, text="Threshold").pack()
        self.threshold_entry = Entry(self.param_frame, width=5)
        self.threshold_entry.pack()
        self.threshold_entry.insert(0, "2")  

        Label(self.param_frame, text="Erosion").pack()
        self.erosion_entry = Entry(self.param_frame, width=5)
        self.erosion_entry.pack()
        self.erosion_entry.insert(0, "5")  

        Label(self.param_frame, text="Dilation").pack()
        self.dilation_entry = Entry(self.param_frame, width=5)
        self.dilation_entry.pack()
        self.dilation_entry.insert(0, "0")  


        Label(self.param_frame, text="Min Pixels").pack()
        self.min_pixels_entry = Entry(self.param_frame, width=5)
        self.min_pixels_entry.pack()
        self.min_pixels_entry.insert(0, "90")  

        Label(self.param_frame, text="Max Pixels").pack()
        self.max_pixels_entry = Entry(self.param_frame, width=5)
        self.max_pixels_entry.pack()
        self.max_pixels_entry.insert(0, "6000")  

        Label(self.param_frame, text="Aspect Ratio Min").pack()
        self.aspect_ratio_min_entry = Entry(self.param_frame, width=5)
        self.aspect_ratio_min_entry.pack()
        self.aspect_ratio_min_entry.insert(0, "0")  

        Label(self.param_frame, text="Aspect Ratio Max").pack()
        self.aspect_ratio_max_entry = Entry(self.param_frame, width=5)
        self.aspect_ratio_max_entry.pack()
        self.aspect_ratio_max_entry.insert(0, "1")  

        Label(self.param_frame, text="Major Axis Min").pack()
        self.major_axis_min_entry = Entry(self.param_frame, width=5)
        self.major_axis_min_entry.pack()
        self.major_axis_min_entry.insert(0, "0")  

        Label(self.param_frame, text="Minor Axis Max").pack()
        self.minor_axis_max_entry = Entry(self.param_frame, width=5)
        self.minor_axis_max_entry.pack()
        self.minor_axis_max_entry.insert(0, "500")  


        # Existing canvas and scrollbar setup remains unchanged
        self.canvas = Canvas(self.root, bg="white")
        self.canvas.pack(fill=BOTH, expand=True)

        hbar = Scrollbar(self.root, orient=HORIZONTAL)
        hbar.pack(side="bottom", fill="x")
        hbar.config(command=self.canvas.xview)

        vbar = Scrollbar(self.root, orient=VERTICAL)
        vbar.pack(side="right", fill="y")
        vbar.config(command=self.canvas.yview)

        # use mouse wheel to scoll through image
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

      # use arrows to scoll through image
        root.bind("<Left>", lambda e: self.scroll_canvas(-2, 0))
        root.bind("<Right>", lambda e: self.scroll_canvas(2, 0))
        root.bind("<Up>", lambda e: self.scroll_canvas(0, -2))
        root.bind("<Down>", lambda e: self.scroll_canvas(0, 2))

    def scroll_canvas(self, dx, dy):
        # Adjust the scroll according to dx and dy
        self.canvas.xview_scroll(dx, "units")
        self.canvas.yview_scroll(dy, "units")

    def prepare_for_lithotag_detection(self):
        # Show a message box to instruct the user
        self.load_and_process_image_for_lithotag_detection()
        self.canvas.bind("<Button-1>", self.on_canvas_click_for_lithotag_detection)

    def load_and_process_image_for_lithotag_detection(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        messagebox.showinfo("Background", "Please click on the Background when image was loaded.")
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.lab = self.correct_illumination(self.original_image)
            self.original_image1 = enhance_contrast(self.image_path)
            self.display_image(self.original_image)

    def correct_illumination(self, img):
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Split the channels
        l_channel, a_channel, b_channel = cv2.split(lab) 
        clahe = cv2.createCLAHE(clipLimit=13, tileGridSize=(4, 4))
        l_channel_eq = clahe.apply(l_channel)
        # Merge the channels and convert back to RGB color space
        corrected_img = cv2.merge((l_channel_eq, a_channel, b_channel))
        return corrected_img

    def on_canvas_click_for_lithotag_detection(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.click_count += 1
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        img_lab = self.lab
        color1 = img_lab[int(canvas_y), int(canvas_x)]

        # Handle the first click - Set background color or another action
        if self.click_count == 1:
            print('clicked on background')# Update instructions or perform the action for the first click
            self.background_color = color1
            print(f"Background color: {self.background_color}")
            messagebox.showinfo("Dot1", "Please click on dot1.")
        elif self.click_count == 2:
            print('clicked on first dot')  # Update instructions or perform the action for the first click
            self.dot_colors.append(color1)
            messagebox.showinfo("Dot2", "Please click on dot2.")
            # Perform action for the second click
        elif self.click_count == 3:
            print('clicked on second dot')  # Update instructions or perform the action for the first click
            self.dot_colors.append(color1)
            print(f"Dot colors: {self.dot_colors}")
            self.process_colors()
            self.ready_for_next_step = True

        elif self.ready_for_next_step:
            # User clicked to indicate ready for next step
            self.ready_for_next_step = False
            # Perform any actions needed for the next step
            self.ask_image_choice()
            # Reset click count if needed and avoid executing the next elif
            self.click_count = 3
            return
            # Perform action for the third click
        elif self.click_count == 4:
            # Perform final action or processing after the fourth click
            self.click_count = 0  # Reset click count if the process should repeat
            self.detect_and_highlight_object(canvas_x, canvas_y)

    def load_image(self):
        if self.original_image is not None:
            print("Image loaded successfully")
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image1 = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.original_image = self.original_image1
            self.height, self.width, _ = self.original_image.shape
            self.original_image = self.original_image[:, :, :3]  # Keep only the first three channels
            self.lab = self.correct_illumination(self.original_image)
 
            self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(self.original_image))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            if self.first_image is None:
                self.first_image = self.original_image.copy()  # Store the first image
                # Convert the image to grayscale
                self.first_image = cv2.cvtColor(self.first_image, cv2.COLOR_RGB2GRAY)

    def ask_image_choice(self):
        user_choice = messagebox.askyesno("Image Selection", "Do you want to use the same image?")
        if user_choice:
            print("Using the same image.")
            self.update_canvas(self.original_image)
        else:
            print("Loading a new image.")
            self.load_image()  # Load and display the new image

        messagebox.showinfo("Action Required", "Please click on the centre of the lithotag.")

    def process_colors(self):

        # Convert image to lab color space
        img_lab = self.lab
        self.height, self.width, _ = self.original_image.shape

        # Initialize mask
        mask = np.zeros((self.height, self.width), dtype=bool)
        dot_colors = self.dot_colors
        background_color_lab = self.background_color
        std_dev = np.std(img_lab)
        threshold = std_dev / 1.2
        threshold1 = std_dev / 1.2 # dark field images
        # threshold1 = std_dev / 4 # brightfield field images

        for color_lab in dot_colors:
            img_lab = np.array(img_lab, dtype=np.float32)
            color_lab = np.array(color_lab, dtype=np.float32)
            # Square the threshold to avoid taking the square root
            squared_threshold = threshold ** 2

            # Compute the squared Euclidean distance and compare with squared threshold
            squared_distance = np.sum((img_lab - color_lab) ** 2, axis=2)
            mask |= squared_distance < squared_threshold

        # Check and mark the background in the mask
        squared_distance_background = np.sum((img_lab - background_color_lab) ** 2, axis=2)
        background_mask = squared_distance_background < (threshold1 ** 2)

        # Create a result image highlighting similar colors
        result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        result[mask & ~background_mask] = [255, 255, 255]  # Dots in white
        result[background_mask] = [0, 0, 0]  # Background in black

        # # Apply morphological operations
        kernel1 = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel1)

        # Store the result image in the attribute
        self.processed_image = result
        self.processed_image1 = result[:, :, 0]
        # Update the canvas image
        self.update_canvas(result)

    def update_canvas(self, new_image):
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(new_image))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def detect_and_highlight_object(self, canvas_x, canvas_y):

        # Convert the image to grayscale
        if self.original_image is not None:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.gray_image_eq = cv2.equalizeHist(self.gray_image)
            binary = cv2.adaptiveThreshold(self.gray_image_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                           2)
            # define the kernel
            kernel = np.ones((3, 3), np.uint8)
            # erode the image
            binary = cv2.erode(binary, kernel,
                               iterations=1)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.pointPolygonTest(contour, (canvas_x, canvas_y), False) >= 0:
                    x1, y1, w, h = cv2.boundingRect(contour)
                    self.highlight_selected_area(x1, y1, w, h)
                    break

        kernel = KernelGeneration(self.selected_region_gray)
        

    def highlight_selected_area(self, x1, y1, w, h):
        # Create a new kernel dictionary
        params={}
        params['ImageThreshold'] = 95  # Default value
        params['SubtractBackground'] = 1
        params['AnnotationColor'] = (0, 0, 255)
        params['CheckSum'] = 0
        params['ImageThreshold'] = 0

        # This method demonstrates drawing a rectangle around a detected object
        self.canvas.create_rectangle(x1, y1, x1+w, y1+h, outline="red", width=2)
        x2, y2 = x1 + w, y1 + h
        self.rect_coords = (x1, y1, x2, y2)
        selected_region = self.gray_image[y1:y2, x1:x2].copy()
        # Create a new kernel dictionary
        kernel = {}
        InputImage = self.gray_image
        vecLithotags = []

        # Perform Multi-Template Matching
        listTemplate = [('LithoTag', selected_region)]
        Hits = matchTemplates(listTemplate, InputImage, score_threshold=0.60, method=cv2.TM_CCOEFF_NORMED,
                              maxOverlap=0)

        currLithotag = {}
 
        boxThickness = 2
        boxColor = (255, 0, 0)
        labelColor = (255, 0, 0)
        labelScale = 0.5
        showCenter = True
        centerColor = (255, 0, 0)
        centerRadius = 3

        if InputImage.ndim == 2:
            outImage = cv2.cvtColor(InputImage, cv2.COLOR_GRAY2RGB)
        else:
            outImage = InputImage.copy()

        # Invert the image
        outImage = cv2.bitwise_not(outImage)

        for _, row in Hits.iterrows():
            x, y, w, h = row['BBox']
            cv2.rectangle(outImage, (x, y), (x + w, y + h), color=boxColor, thickness=boxThickness)

            cv2.putText(outImage, text=row['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA)

            if showCenter:
                # Calculate and draw the center of the bounding box
                centerX, centerY = x + w // 2, y + h // 2
                cv2.circle(outImage, (centerX, centerY), centerRadius, color=centerColor, thickness=-1)  # Filled circle

            # Create a new dictionary for the current tag
            currLithotag = {}
            currLithotag['Image'] = InputImage[y:y + h, x:x + w]
            currLithotag['Centroid'] = [centerX, centerY]
            currLithotag['BBox'] = (x, y, w, h)

            # Append the current tag to the list
            vecLithotags.append(currLithotag)

        # Perform Multi-Template Matching
        listTemplate = [('LithoTag', selected_region)]
        Hits = matchTemplates(listTemplate, InputImage, score_threshold=0.60, method=cv2.TM_CCOEFF_NORMED,
                              maxOverlap=0)

        # After vecLithotags has been populated with the bounding boxes
        if len(vecLithotags) > 1:
            min_distance = float('inf')
            closest_pair = None

            # Compare each pair of centers
            for i in range(len(vecLithotags)):
                for j in range(i + 1, len(vecLithotags)):
                    distance = calculate_distance(vecLithotags[i]['Centroid'],
                                                  vecLithotags[j]['Centroid'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (vecLithotags[i]['Centroid'], vecLithotags[j]['Centroid'])
                        closest_pair_indices = (i, j)

            # Print or use the closest pair and their distance
            if closest_pair:
                print("Closest Pair:", closest_pair)
                print("Minimum Distance:", min_distance)
                kernel_width_px = min_distance / 8.2
                kernel_height_px = min_distance / 8.2
                print("kernel width pixels", kernel_width_px)
                print("kernel height pixels", kernel_height_px)

            m = round((kernel_width_px - w) / 2)
            n = round((kernel_height_px - h) / 2)

            selected_region_gray = InputImage[y - n:y + h + n, x - m:x + w + m]

        listTemplate = [('LithoTag', selected_region_gray)]
        Hits1 = matchTemplates(listTemplate, InputImage, score_threshold=0.60, method=cv2.TM_CCOEFF_NORMED,
                               maxOverlap=0)
        # Display the results
        print("Found {} hits".format(len(Hits1.index)))

        for _, row in Hits1.iterrows():
            x, y, w, h = row['BBox']
            cv2.rectangle(outImage, (x, y), (x + w, y + h), color=boxColor, thickness=boxThickness)

            cv2.putText(outImage, text=row['TemplateName'], org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA)

            if showCenter:
                # Calculate and draw the center of the bounding box
                centerX, centerY = x + w // 2, y + h // 2
                cv2.circle(outImage, (centerX, centerY), centerRadius, color=centerColor, thickness=-1)  # Filled circle

            # Perform operations only on the closest pair
            if closest_pair_indices:
                i, j = closest_pair_indices

                object1 = vecLithotags[i]['Image']
                object2 = vecLithotags[j]['Image']

                score_top_right = compare_quadrants_within_object(object1, object2, 2)
                score_top_left = compare_quadrants_within_object(object1, object2, 1)

                print("score_top_right = ", score_top_right)
                print("score_top_left = ", score_top_left)

                # Determine which match is better
                better_match = "Top Right" if score_top_right > score_top_left else "Top Left"

                if better_match == "Top Right":
                    print("tags are horizontal to each other")

                    point11 = vecLithotags[i]['Centroid']
                    point22 = vecLithotags[j]['Centroid']

                    if point22[0] > point11[0]:
                        point1 = vecLithotags[i]['Centroid']
                        point2 = vecLithotags[j]['Centroid']
                    else:
                        point1 = vecLithotags[j]['Centroid']
                        point2 = vecLithotags[i]['Centroid']

                    # Calculate the angle in radians
                    theta = math.atan2(point2[1] - point1[1], point2[0] - point1[0])

                    # Convert the angle from radians to degrees
                    theta_degrees1 = math.degrees(theta)
                    if theta_degrees1 < 0:
                        theta_degrees1 = -theta_degrees1
                    # elif theta_degrees1>90 and  theta_degrees1<180:
                    #     theta_degrees1= 180 - theta_degrees1
                    if theta_degrees1 > 70:
                        theta_degrees1 = 90 - theta_degrees1
                    print("theta_degrees is ", theta_degrees1)

                    x, y, w, h = row['BBox']

                    center = x + w // 2, y + h // 2
                    width = w // 2
                    height = h // 2
                    degrees = theta_degrees1

                    # Calculate rotated rectangle coordinates
                    pt1 = [center[0] - width, center[1] - height]
                    pt2 = [center[0] - width, center[1] + height]
                    pt3 = [center[0] + width, center[1] + height]
                    pt4 = [center[0] + width, center[1] - height]

                    rectangle = [pt1, pt2, pt3, pt4, pt1]

                    rectangle_rotated = [rotate(center, pt, math.radians(degrees)) for pt in rectangle]

                    # Extract x and y coordinates
                    x_coords = [point[0] for point in rectangle_rotated]
                    y_coords = [point[1] for point in rectangle_rotated]

                    # Calculate the rough dimensions of the image based on the points
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)

                    # Assuming the image is centered around these points
                    center = (min(x_coords) + width // 2, min(y_coords) + height // 2)

                    # Rotate each point by 45 degrees around the center

                    rotated_points = [rotate_point(point, center, degrees) for point in rectangle_rotated]
                    x_rotated, y_rotated, w_rotated, h_rotated = get_bounding_box(rotated_points)


                    rectangle_rotated = np.array(rectangle_rotated, np.int32).reshape((-1, 1, 2))

                    
                    params['angle'] = theta_degrees1
                    self.rotated_rectangles.append(rectangle_rotated)

                else:
                    print("tags are vertical to each other")

                    point11 = vecLithotags[i]['Centroid']
                    point22 = vecLithotags[j]['Centroid']

                    if point22[1] > point11[1]:
                        point1 = vecLithotags[i]['Centroid']
                        point2 = vecLithotags[j]['Centroid']
                    else:
                        point1 = vecLithotags[j]['Centroid']
                        point2 = vecLithotags[i]['Centroid']

                    # Calculate the angle in radians
                    theta = math.atan2(point1[1] - point2[1], point2[0] - point1[0])
                    # Convert the angle from radians to degrees
                    theta_degrees2 = 90 + math.degrees(theta)

                    if theta_degrees2 < 0:
                        theta_degrees2 = - theta_degrees2
                    # elif theta_degrees2>100:
                    #     theta_degrees2 = 180 - theta_degrees2
                    if theta_degrees2 > 70:
                        theta_degrees2 = 90 - theta_degrees2
                    print("theta_degrees is ", theta_degrees2)

                    x, y, w, h = row['BBox']

                    center = x + w // 2, y + h // 2
                    width = w // 2
                    height = h // 2
                    degrees = theta_degrees2

                    # Calculate rotated rectangle coordinates
                    pt1 = [center[0] - width, center[1] - height]
                    pt2 = [center[0] - width, center[1] + height]
                    pt3 = [center[0] + width, center[1] + height]
                    pt4 = [center[0] + width, center[1] - height]

                    rectangle = [pt1, pt2, pt3, pt4, pt1]

                    rectangle_rotated = [rotate(center, pt, math.radians(degrees)) for pt in rectangle]
                    # Draw the rotated bounding box on the image
                    # Extract x and y coordinates
                    x_coords = [point[0] for point in rectangle_rotated]
                    y_coords = [point[1] for point in rectangle_rotated]

                    # Calculate the rough dimensions of the image based on the points
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)

                    # Assuming the image is centered around these points
                    center = (min(x_coords) + width // 2, min(y_coords) + height // 2)

                    # Rotate each point by 45 degrees around the center

                    rotated_points = [rotate_point(point, center, degrees) for point in rectangle_rotated]
                    x_rotated, y_rotated, w_rotated, h_rotated = get_bounding_box(rotated_points)

                    rectangle_rotated = np.array(rectangle_rotated, np.int32).reshape((-1, 1, 2))
                    self.rotated_rectangles.append(rectangle_rotated)

                    # Draw the rotated bounding box on the image
                    cv2.polylines(InputImage, [rectangle_rotated], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Display the image with the bounding box

                   
                    params['angle'] = theta_degrees2

            # Calculate and draw the center of the bounding box
            centerX_rotated, centerY_rotated = x_rotated + w_rotated // 2, y_rotated + h_rotated // 2
            cv2.circle(outImage, (centerX, centerY), centerRadius, color=centerColor, thickness=-1)  # Filled circle
            # Create a new dictionary for the current tag
            currLithotag1 = {}
            currLithotag1['Image'] = InputImage[y_rotated:y_rotated + h_rotated, x_rotated:x_rotated + w_rotated]
            currLithotag1['Centroid'] = [centerX_rotated, centerY_rotated]
            currLithotag1['BBox'] = (x_rotated, y_rotated, w_rotated, h_rotated)


            # Append the current tag to the list
            self.tag.append(currLithotag1)
            self.selected_region_gray=currLithotag1['Image']
            self.params = params

    def prepare_for_contour_removal(self):
        self.find_and_draw_contours()
        self.canvas.bind("<Button-1>", self.on_canvas_click_for_contour_removal)

    def on_canvas_click_for_contour_removal(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        remove_indices = {idx for idx, contour in enumerate(self.contours) if
                          cv2.pointPolygonTest(contour, (canvas_x, canvas_y), False) >= 0}
        self.contours = [contour for idx, contour in enumerate(self.contours) if idx not in remove_indices]
        self.redraw_image()

    def show_binary_image(self):

        if hasattr(self, 'original_image'):
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            threshold_value = int(self.threshold_entry.get())
            erosion_value = int(self.erosion_entry.get())
            dilation_value = int(self.dilation_entry.get())
            binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21,
                                           threshold_value)
            # Apply erosion and dilation
            kernel1 = np.ones((erosion_value, erosion_value), np.uint8)
            binary = cv2.erode(binary, kernel1, iterations=1)
            kernel2 = np.ones((dilation_value, dilation_value), np.uint8)
            binary = cv2.dilate(binary, kernel2, iterations=1)

            # Check if the binary image should be inverted based on the new checkbox
            if self.invert_binary_var.get() == 1:
                binary_image = cv2.bitwise_not(binary)
            else:
                binary_image = binary  # No inversion if the checkbox is not checked

            self.binary_image = binary_image
            # The display image method checks the invert_image_var for displaying the inverted image if needed
            self.display_image(
                cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR))  # Convert binary image to BGR for displaying

            image_height, image_width = binary_image.shape

            coordinates_size = []

            for rectangle in self.rotated_rectangles:
                # Extract the top-left corner of the rectangle (assuming it's the first coordinate)
                # and convert to a list if it's a numpy array
                first_coordinate = np.array(rectangle[0]).flatten().tolist()
                # Convert each float value to integer and append the fixed values (169, 169)
                new_format = [int(first_coordinate[0]), int(first_coordinate[1]), 169, 169]
                coordinates_size.append(new_format)

            # Now coordinates_size is in the desired format
            print(coordinates_size)


            small = 1
            big_sf = 1.2
            small_sf = 1.4

            scale_factor = big_sf
            for each_bounding_box in coordinates_size:
                corner_point = [each_bounding_box[0], each_bounding_box[1]]
                width, height = each_bounding_box[2], each_bounding_box[3]
                # Calculate the corner points after applying a scale factor to remove the lithotags
                point1 = [int(np.floor(corner_point[0] - width * (scale_factor - 1))),
                          int(np.floor(corner_point[1] - height * (scale_factor - 1)))]
                point2 = [int(np.ceil(corner_point[0] + width * scale_factor)),
                          int(np.ceil(corner_point[1] + height * scale_factor))]
                cv2.rectangle(binary_image, point1, point2, (0, 0, 0), cv2.FILLED)
            if small == 1:
                # Calculate the points of the mini lithotags, would require the first two lithotags
                scale_factor = small_sf
                bigtag_x_spacing = abs(coordinates_size[0][0] - coordinates_size[1][0])
                bigtag_y_spacing = abs(coordinates_size[0][1] - coordinates_size[1][1])
                bigtag_spacing = max(bigtag_x_spacing, bigtag_y_spacing)
                smalltag_spacing = int(np.ceil(bigtag_spacing / 5))
                smalltag_w = int(np.ceil(abs(coordinates_size[0][2]) / 5))
                smalltag_h = int(np.ceil(abs(coordinates_size[0][3]) / 5))
                # generate all the coordinates of the corners of the small tags:
                bigtag1 = min(coordinates_size, key=lambda coord: coord[0] + coord[1])
                bigtag1_centre = [int(np.floor(bigtag1[0] + bigtag1[2] / 2)),
                                  int(np.floor(bigtag1[1] + bigtag1[3] / 2))]
                centre_point = bigtag1_centre
                point1 = [int(np.floor(centre_point[0] - smalltag_w / 2) - smalltag_w * (scale_factor - 1)),
                          int(np.floor(centre_point[1] - smalltag_h / 2) - smalltag_h * (scale_factor - 1))]
                point2 = [int(np.floor(centre_point[0] + smalltag_w / 2) + smalltag_w * (scale_factor - 1)),
                          int(np.floor(centre_point[1] + smalltag_h / 2) + smalltag_h * (scale_factor - 1))]
                initial_x1 = point1[0]
                initial_x2 = point2[0]
                while point2[1] < image_height:
                    cv2.rectangle(binary_image, point1, point2, (0, 0, 0), cv2.FILLED)
                    point1[0] += smalltag_spacing
                    point2[0] += smalltag_spacing
                    if point2[0] > image_width:
                        point1[0] = initial_x1
                        point2[0] = initial_x2
                        point1[1] += smalltag_spacing
                        point2[1] += smalltag_spacing
                    point1[0] = int(point1[0])
                    point1[1] = int(point1[1])
                    point2[0] = int(point2[0])
                    point2[1] = int(point2[1])


            self.binary_image = binary_image
            self.display_image(
                cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR))  # Convert binary image to BGR for displaying



    def find_and_draw_contours(self):
        if hasattr(self, 'original_image'):

            self.display_image(cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR))  # Convert binary image to BGR for displaying
            contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Read values from UI
            min_pixels = int(self.min_pixels_entry.get())
            max_pixels = int(self.max_pixels_entry.get())
            aspect_ratio_max = float(self.aspect_ratio_max_entry.get())
            aspect_ratio_min = float(self.aspect_ratio_min_entry.get())
            major_axis_min = float(self.major_axis_min_entry.get())
            minor_axis_max = float(self.minor_axis_max_entry.get())

            # Update filter criteria based on UI values
            self.contours = filter_contours_by_criteria(contours, min_pixels, max_pixels, aspect_ratio_min,aspect_ratio_max,
                                                        major_axis_min, minor_axis_max)
            self.redraw_image()

    def redraw_image(self):
        image_with_contours = self.original_image.copy()
        for contour in self.contours:
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)
        self.display_image(image_with_contours)

    def display_image(self, img):
        # Check if the image should be inverted
        if self.invert_image_var.get() == 1:
            img = cv2.bitwise_not(img)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(self.image_on_canvas))


    def on_canvas_click(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        remove_indices = {idx for idx, contour in enumerate(self.contours) if cv2.pointPolygonTest(contour, (canvas_x, canvas_y), False) >= 0}
        self.contours = [contour for idx, contour in enumerate(self.contours) if idx not in remove_indices]
        self.redraw_image()

    def on_mouse_wheel(self, event):
        if event.state & 0x0001:  # Shift key is down
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:  # No modifier keys are pressed
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")



    def plot_and_save_contours_rgb(self):
        original_dir = os.path.dirname(self.image_path)
        image_prefix = os.path.basename(self.image_path).split('.')[0]

        parent_dir = os.path.join(original_dir, 'contour_plots_rgb')
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        # Assuming the images have a file extension such as .bmp
        image_path = os.path.join(original_dir, f"{image_prefix}.bmp")
        new_image = cv2.imread(image_path)

        output_dir = os.path.join(parent_dir, f'contour_plots_rgb_{image_prefix}')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        image_for_contour_map = new_image.copy()

        for i, contour in enumerate(self.contours):
            cv2.drawContours(image_for_contour_map, [contour], -1, (0, 0, 255), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(image_for_contour_map, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)

            cv2.imwrite(os.path.join(output_dir, f"{image_prefix}_contour_map.png"), image_for_contour_map)

            if len(contour) < 5:
                continue

            start_time = time.time()
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            (xc, yc) = center
            (d1, d2) = axes
            a, b = max(d1, d2) / 2, min(d1, d2) / 2
            theta = math.radians(angle)

            mask = np.zeros(new_image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val, std_dev = cv2.meanStdDev(new_image, mask=mask)

            # Extract RGB values within the contour region
            rgb_values = new_image[mask == 255]

            end_time = time.time()
            print(f"Elapsed time for sampling RGB values: {end_time - start_time:.2f} seconds")

            # Save RGB values to CSV file
            csv_file_path = os.path.join(output_dir, f"{image_prefix}_contour_{i + 1}_rgb_values.csv")
            fieldnames = ['R', 'G', 'B']

            with open(csv_file_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for rgb_value in rgb_values:
                    writer.writerow({'R': rgb_value[2], 'G': rgb_value[1], 'B': rgb_value[0]})

            print(f"RGB values for contour {i + 1} have been saved to {csv_file_path}")

        print("All contours have been processed and saved.")
        # Show a message box indicating completion (assuming you have a suitable GUI library imported)
        messagebox.showinfo("Process Complete",
                            "RGB values for all contours have been successfully saved to individual CSV files.")

def KernelGeneration(selected_regions):
    kernel = {}

    # Initialize properties using keys
    kernel['CoordLocsX'] = []
    kernel['CoordLocsY'] = []
    kernel['CheckLocsX'] = []
    kernel['CheckLocsY'] = []
    kernel['pCentre'] = (0.0, 0.0)  # example value for center point
    kernel['Shape'] = "Hexagon"
    kernel['NoRings'] = 3

    KernelImage = selected_regions
    # plt.imshow(KernelImage)
    # plt.show()

    kernel['Height'] = KernelImage.shape[0]
    kernel['Width'] = KernelImage.shape[1]

    # Normalize and convert to floating point
    # KernelImage = KernelImage.astype(np.float32) / cv2.sumElems(KernelImage)[0]

    # Allocate to Kernel['Image'] for further processing
    kernel['Image'] = KernelImage.copy()

    # Initialize Kernel object
    pi = math.pi

    # Calculate coordinate locations
    for colIdx in range(1, kernel['NoRings'] + 1):
        for rowIdx in range(1, kernel['NoRings'] + 1):
            # Coord Locs X
            pCentre_x = -math.cos(pi / 6) * colIdx * 2
            pCentre_y = 2 * math.sin(pi / 6) * colIdx - 2 * (kernel['NoRings'] + 1 - rowIdx)
            pCentre_x = pCentre_x * kernel['Width'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Width'] / 2
            pCentre_y = pCentre_y * kernel['Height'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Height'] / 2
            kernel['CoordLocsX'].append((pCentre_x, pCentre_y))

            # Coord Locs Y
            pCentre_x = math.cos(pi / 6) * colIdx * 2
            pCentre_y = 2 * math.sin(pi / 6) * colIdx - 2 * (kernel['NoRings'] + 1 - rowIdx)
            pCentre_x = pCentre_x * kernel['Width'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Width'] / 2
            pCentre_y = pCentre_y * kernel['Height'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Height'] / 2
            kernel['CoordLocsY'].append((pCentre_x, pCentre_y))

    # Check locations
    colIdx = 1
    rowIdx = 1
    for bitIdx in range(1, kernel['NoRings'] * (kernel['NoRings'] - 1) // 2 + 1):
        # Check Locs X
        pCentre_x = -math.cos(pi / 6) * colIdx * 2
        pCentre_y = math.sin(pi / 6) * colIdx * 2 + 2 * rowIdx
        pCentre_x = pCentre_x * kernel['Width'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Width'] / 2 - 1
        pCentre_y = pCentre_y * kernel['Height'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel['Height'] / 2
        kernel['CheckLocsX'].append((pCentre_x, pCentre_y))

        # Check Locs Y
        pCentre_x = math.cos(pi / 6) * colIdx * 2  # Changed -math.cos to math.cos
        pCentre_y = math.sin(pi / 6) * colIdx * 2 + 2 * rowIdx  # Same as original
        pCentre_x = pCentre_x * kernel['Width'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel[
            'Width'] / 2 - 1
        pCentre_y = pCentre_y * kernel['Height'] / ((kernel['NoRings'] - 1) * 4 + 6) + kernel[
            'Height'] / 2
        kernel['CheckLocsY'].append((pCentre_x, pCentre_y))

        if (colIdx + rowIdx) > kernel['NoRings'] - 1:
            colIdx = 1
            rowIdx += 1
        else:
            colIdx += 1
    return kernel

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def extract_quadrant(image, bbox, side):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    if side == 1:  # Left side
        return image[y:y + h, x:cx]
    elif side == 2:  # Right side
        return image[y:y + h, cx:x + w]


def compare_quadrants_within_object(object1, object2, quadrant):
    quad1 = extract_quadrant(object1, (0, 0, object1.shape[1], object1.shape[0]), quadrant)
    quad2 = extract_quadrant(object2, (0, 0, object2.shape[1], object2.shape[0]), quadrant)

    # Resize for comparison if necessary
    if quad1.shape != quad2.shape:
        quad2 = cv2.resize(quad2, (quad1.shape[1], quad1.shape[0]))

    # Flatten the image arrays
    flat_quad1 = quad1.flatten()
    flat_quad2 = quad2.flatten()

    # Calculate the Pearson Correlation Coefficient
    correlation, _ = pearsonr(flat_quad1, flat_quad2)
    correlation = abs(correlation)
    return correlation

def rotate(origin, point, angle):
    angle = -angle
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def get_bounding_box(rectangles_rotated):
    # Assuming rectangles_rotated is a list of tuples (x, y)

    # Separate the x and y coordinates
    x_coords = [pt[0] for pt in rectangles_rotated]
    y_coords = [pt[1] for pt in rectangles_rotated]

    # Calculate x, y, w, h
    x = min(x_coords)
    y = min(y_coords)
    w = max(x_coords) - x
    h = max(y_coords) - y

    return x, y, w, h


def rotate_point(point, center, angle):
    """
    Rotate a point counterclockwise by a given angle around a given center.

    :param point: The coordinates of the point to rotate (x, y).
    :param center: The coordinates of the center point of rotation (x_center, y_center).
    :param angle: The angle in degrees to rotate the point.
    :return: The coordinates of the rotated point.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Translate point to origin
    x, y = point
    x_center, y_center = center
    x -= x_center
    y -= y_center

    # Rotate point
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translate point back
    x_new += x_center
    y_new += y_center

    return (int(round(x_new)), int(round(y_new)))

def rotate(origin, point, angle):
    angle = -angle
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

if __name__ == "__main__":
    root = Tk()
    root.title("Nanowire Detection")
    app = ImageScrollApp(root)
    root.geometry("2000x1500")
    root.mainloop()
