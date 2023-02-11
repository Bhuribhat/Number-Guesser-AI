import os
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import ImageGrab
from constant import *


# create a folder if not exist
if not os.path.exists('./saved_images'):
    os.mkdir('./saved_images')


def get_label_name():
    label = image_name.get()
    if label == '':
        label = len(os.listdir('./saved_images'))
        label = f"unlabel_{1 + (label // 2)}"
    return [f"gray_{label}.png", f"resized_{label}.png"]


def draw_line(event):
    x1, y1 = (event.x - MARKER_SIZE), (event.y - MARKER_SIZE)
    x2, y2 = (event.x + MARKER_SIZE), (event.y + MARKER_SIZE)
    canvas.create_rectangle(x1, y1, x2, y2, fill=WHITE, outline=WHITE)


def erase_line(event):
    x1, y1 = (event.x - MARKER_SIZE), (event.y - MARKER_SIZE)
    x2, y2 = (event.x + MARKER_SIZE), (event.y + MARKER_SIZE)
    canvas.create_rectangle(x1, y1, x2, y2, fill=DARK, outline=DARK)


def erase_canvas():
    canvas.delete("all")
    status.configure(text="Please draw a number!", fg="deepskyblue")
    image_name.delete(0, END)


def save_image():
    x1 = canvas.winfo_rootx() + canvas.winfo_x()
    y1 = canvas.winfo_rooty() + canvas.winfo_y()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()

    fig = plt.figure(figsize=(12, 8))
    name1, name2 = get_label_name()

    # save original image 500 x 500
    image = ImageGrab.grab((x1, y1, x2, y2))
    image = np.array(image)
    
    # convert white to gray scale image
    fig.add_subplot(1, 2, 1)
    gray_image = tf.image.rgb_to_grayscale(image)
    cv2.imwrite(f'./saved_images/{name1}', gray_image.numpy())
    plt.imshow(gray_image.numpy(), cmap='gray')
    plt.title(f"500x500 of {image_name.get()}")

    # save gray scale image 28 x 28
    fig.add_subplot(1, 2, 2)
    resize_image = tf.image.resize(gray_image, (28, 28), method='nearest')
    cv2.imwrite(f'./saved_images/{name2}', resize_image.numpy())
    plt.imshow(resize_image.numpy(), cmap='gray')
    plt.title(f"28x28 of {image_name.get()}")
    plt.show()


def predict():
    x1 = canvas.winfo_rootx() + canvas.winfo_x()
    y1 = canvas.winfo_rooty() + canvas.winfo_y()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()

    image = ImageGrab.grab((x1, y1, x2, y2))
    image = np.array(image)

    # image = tf.keras.utils.array_to_img(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (28, 28), method='nearest')

    # load model: input shape = (1, 28, 28)
    image = tf.reshape(image, (1, 28, 28))
    model = tf.keras.models.load_model('./assets/number_model.h5')

    # Probability of all numbers
    predictions = model.predict(image)
    top_three = np.argsort(predictions[0])[-3:]
    display_text = f"Prediction = {top_three[-1]}\n"

    # print("Probability", predictions[0])
    for idx, value in enumerate(top_three[::-1]):
        percent = predictions[0][value]
        print(f"{idx + 1}. Predict {value} with probability = {percent}%")

    # Prediction: np.argmax(predictions[0])
    percent = predictions[0][top_three[-1]] * 100
    display_text += f"\nProbabillity = {percent:.0f}%"
    status.configure(text=display_text, fg="deepskyblue")


if __name__ == '__main__':
    ROOT = tk.Tk()
    ROOT.title("Number Guesser")
    ROOT.resizable(False, False)
    ROOT.configure(background=DKBLUE)
    ROOT.bind('<Escape>', lambda e: ROOT.quit())

    status = Label(
        ROOT, text="Please draw a number!", 
        padx=10, font=20, fg="deepskyblue", bg=DKBLUE
    )
    status.pack(pady=20)

    # First Frame
    draw_frame = Frame(ROOT, pady=10, bg=DKBLUE)
    draw_frame.pack()

    canvas = Canvas(draw_frame, bg=DARK, width=WIDTH, height=HEIGHT)
    canvas.pack(padx=15)

    # Second Frame
    input_frame = Frame(ROOT, pady=10, bg=DKBLUE)
    input_frame.pack()

    # user input button and entry
    text = StringVar()
    CustomLabel(input_frame, text="Label").grid(row=0, column=0, pady=10, sticky="news")
    image_name = CustomEntry(input_frame, textvariable=text)
    image_name.grid(row=0, column=1, columnspan=2, sticky="ew")

    predict_button = HooverButton(input_frame, text="Predict", command=predict)
    predict_button.grid(row=1, column=0, padx=15, pady=10, sticky='news')

    save_button = HooverButton(input_frame, text="Save", command=save_image)
    save_button.grid(row=1, column=1, padx=15, pady=10, sticky='news')

    erase_button = HooverButton(input_frame, text="Clear", command=erase_canvas)
    erase_button.grid(row=1, column=2, padx=15, pady=10, sticky='news')

    # Left Click to draw
    canvas.bind("<B1-Motion>", draw_line)

    # Right CLick to erase
    canvas.bind("<B3-Motion>", erase_line)

    ROOT.mainloop()