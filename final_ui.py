import tkinter as tk
from tkinter import filedialog
import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageTk


model_weights_path = 'yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)


def get_ROI_coordinates(image_path):
    frame = cv2.imread(image_path)

    results = model(frame)
    rois_coordinates = []

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = row['name']
        rois_coordinates.append(((x1, y1), (x2, y2), d))

    return rois_coordinates


def stage2(x):
    ls=x<<1
    if x>=128:
        ls+=1
    rs=x>>1
    if x%2==1:
        rs+=128
    a90=(ls^rs)
    a90=a90&85
    a60=x^rs
    a60=a60&170
    return a60+a90
def show_camera_feed(cap, camera_label):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    camera_label.config(image=photo)
    camera_label.image = photo
    camera_label.after(10, lambda: show_camera_feed(cap, camera_label))  # Use lambda to pass arguments

def capture_image(cap):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.save("captured_image.jpg")
    image_filename = "captured_image.jpg"
    image_path = image_filename  # Use the filename as the image path
    rois_coordinates = get_ROI_coordinates(image_path)
    os.remove(image_path)
    minx=frame.shape[1]
    miny=frame.shape[0]
    maxx=0
    maxy=0
    for i, roi in enumerate(rois_coordinates):
        (x1, y1), (x2, y2), label = roi
        print(x1," ",y1," ",x2," ",y2," ",label)
        if label=='person':
            minx=x1
            miny=y1
            maxx=x2
            maxy=y2
    
    y=maxx-minx
    x=maxy-miny
    z=3

    print("maxx"," ",maxx,"minx"," ",minx)
    print("maxy"," ",maxy,"miny"," ",miny)

    if y>x:
         if miny>=(y-x):
              miny=miny-y+x
         else:
              miny=0
              maxy=y
    else:
         if minx>=(x-y):
              minx=minx-x+y
         else:
              minx=0
              maxx=x

    y=maxx-minx
    x=maxy-miny
    print(x," ",y," ",z)

    temp=np.zeros((x,y,z),dtype=np.uint8)

    for i in range(0,x):
         for j in range(0,y):
                temp[i,j,0]=frame[miny+i,minx+j,0]
                temp[i,j,1]=frame[miny+i,minx+j,1]
                temp[i,j,2]=frame[miny+i,minx+j,2]

    #cv2.imshow("IMAGE",temp)
    #cv2.waitKey(0)


    temp1=np.zeros((x,y,z),dtype=np.uint8)

    for i in range(0,x):
        for j in range(0,y):
                temp1[(1-i*i+j)%x,(i+1)%y,0]=temp[i,j,0]
                temp1[(1-i*i+j)%x,(i+1)%y,1]=temp[i,j,1]
                temp1[(1-i*i+j)%x,(i+1)%y,2]=temp[i,j,2]

    temp2=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
                temp2[(1-i*i+j)%x,(i+1)%y,0]=temp1[i,j,0]
                temp2[(1-i*i+j)%x,(i+1)%y,1]=temp1[i,j,1]
                temp2[(1-i*i+j)%x,(i+1)%y,2]=temp1[i,j,2]

    temp3=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
                temp3[(1-i*i+j)%x,(i+1)%y,0]=temp2[i,j,0]
                temp3[(1-i*i+j)%x,(i+1)%y,1]=temp2[i,j,1]
                temp3[(1-i*i+j)%x,(i+1)%y,2]=temp2[i,j,2]


    temp3=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
             temp3[i,j,0]=stage2(temp2[i,j,0])
             temp3[i,j,1]=stage2(temp2[i,j,1])
             temp3[i,j,2]=stage2(temp2[i,j,2])

    for i in range(miny,maxy):
         for j in range(minx,maxx):
              frame[i,j]=temp3[i-miny,j-minx]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow("IMG",frame)
    #cv2.waitKey(0)
    cv2.imwrite("roiimage.png",frame)

def cam():
    root.destroy()
    app = tk.Tk()
    app.title("Camera Capture App")

    camera_label = tk.Label(app)
    camera_label.pack(padx=10, pady=10)

    capture_button = tk.Button(app, text="Capture", command=lambda: capture_image(cap))
    capture_button.pack(pady=10)

    def show_camera_feed():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        camera_label.config(image=photo)
        camera_label.image = photo
        camera_label.after(10, show_camera_feed)

    show_camera_feed()
     
    app.mainloop()
    # Release the camera when the application is closed
    cap.release()

def open_file_explorer():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        frame = cv2.imread(file_path)
        # Do further processing with the 'image' variable as needed
        # For example, you can display the image using cv2.imshow()
    else:
         print("ERROR")
    image_path=file_path
    rois_coordinates = get_ROI_coordinates(image_path)
    minx=frame.shape[1]
    miny=frame.shape[0]
    maxx=0
    maxy=0
    for i, roi in enumerate(rois_coordinates):
        (x1, y1), (x2, y2), label = roi
        print(x1," ",y1," ",x2," ",y2," ",label)
        if label=='person':
            minx=x1
            miny=y1
            maxx=x2
            maxy=y2
    
    y=maxx-minx
    x=maxy-miny
    z=3

    print("maxx"," ",maxx,"minx"," ",minx)
    print("maxy"," ",maxy,"miny"," ",miny)

    if y>x:
         if miny>=(y-x):
              miny=miny-y+x
         else:
              miny=0
              maxy=y
    else:
         if minx>=(x-y):
              minx=minx-x+y
         else:
              minx=0
              maxx=x

    y=maxx-minx
    x=maxy-miny
    print(x," ",y," ",z)

    temp=np.zeros((x,y,z),dtype=np.uint8)

    for i in range(0,x):
         for j in range(0,y):
                temp[i,j,0]=frame[miny+i,minx+j,0]
                temp[i,j,1]=frame[miny+i,minx+j,1]
                temp[i,j,2]=frame[miny+i,minx+j,2]

    #cv2.imshow("IMAGE",temp)
    #cv2.waitKey(0)


    temp1=np.zeros((x,y,z),dtype=np.uint8)

    for i in range(0,x):
        for j in range(0,y):
                temp1[(1-i*i+j)%x,(i+1)%y,0]=temp[i,j,0]
                temp1[(1-i*i+j)%x,(i+1)%y,1]=temp[i,j,1]
                temp1[(1-i*i+j)%x,(i+1)%y,2]=temp[i,j,2]

    temp2=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
                temp2[(1-i*i+j)%x,(i+1)%y,0]=temp1[i,j,0]
                temp2[(1-i*i+j)%x,(i+1)%y,1]=temp1[i,j,1]
                temp2[(1-i*i+j)%x,(i+1)%y,2]=temp1[i,j,2]

    temp3=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
                temp3[(1-i*i+j)%x,(i+1)%y,0]=temp2[i,j,0]
                temp3[(1-i*i+j)%x,(i+1)%y,1]=temp2[i,j,1]
                temp3[(1-i*i+j)%x,(i+1)%y,2]=temp2[i,j,2]


    temp3=np.zeros((x,y,z),dtype=np.uint8)
    for i in range(0,x):
        for j in range(0,y):
             temp3[i,j,0]=stage2(temp2[i,j,0])
             temp3[i,j,1]=stage2(temp2[i,j,1])
             temp3[i,j,2]=stage2(temp2[i,j,2])

    for i in range(miny,maxy):
         for j in range(minx,maxx):
              frame[i,j]=temp3[i-miny,j-minx]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow("IMG",frame)
    #cv2.waitKey(0)
    cv2.imwrite("roiimage.png",frame)


root = tk.Tk()
root.title("IMG ENCR")
cap = cv2.VideoCapture(0)  # Change the index to select a different camera (if available)

b1 = tk.Button(root, text="Use Camera", command=cam)
b1.pack(padx=10, pady=10)

file_button = tk.Button(root, text="Open Image", command=open_file_explorer)
file_button.pack(pady=20)

root.mainloop()