import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import ImageGrab
from tkinter import *

# Constants
DARK = "#1F2937"
WHITE = "#FFFFFF"

DKBLUE = "#111827"
DARKER = "#1F2937"
GRAY   = "#545454"
PURPLE = "#8B5CF6"


class HooverButton(Button):
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self["borderwidth"] = 0
        self["font"] = 7
        self["width"] = 12
        self["fg"] = "white"
        self["bg"] = GRAY
        self["cursor"] = "hand2"
        self["activeforeground"] = "white"
        self["activebackground"] = DARKER
        self["disabledforeground"] = DARKER

        self.bind('<Enter>', lambda e: self.config(background=PURPLE))
        self.bind('<Leave>', lambda e: self.config(background=GRAY))


def draw_line(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_rectangle(x1, y1, x2, y2, fill=WHITE, outline=WHITE)


def erase_line(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_rectangle(x1, y1, x2, y2, fill=DARK, outline=DARK)


def erase_canvas():
    canvas.delete("all")
    status.configure(text="Please draw a number!", fg="deepskyblue")


def rgb_to_gray(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def convert_color(image, old_value, new_value):
    r1, g1, b1 = old_value
    r2, g2, b2 = new_value

    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    image[:,:,:3][mask] = [r2, g2, b2]
    return image


def save_image():
    x1 = canvas.winfo_rootx() + canvas.winfo_x()
    y1 = canvas.winfo_rooty() + canvas.winfo_y()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()

    fig = plt.figure(figsize=(12, 8))

    # save original image 500 x 500
    image = ImageGrab.grab((x1, y1, x2, y2))
    image = np.array(image)
    
    # convert white to gray scale image
    fig.add_subplot(1, 2, 1)
    image = tf.image.rgb_to_grayscale(image)
    cv2.imwrite('./assets/gray.png', image.numpy())
    plt.imshow(image.numpy(), cmap='gray')

    # save gray scale image 28 x 28
    fig.add_subplot(1, 2, 2)
    image = tf.image.resize(image, (28, 28))
    cv2.imwrite('./assets/input.png', image.numpy())
    plt.imshow(image.numpy(), cmap='gray')
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
    image = tf.image.resize(image, (28, 28))
    

    # load model: input shape = (1, 28, 28)
    image = tf.reshape(image, (1, 28, 28))
    model = tf.keras.models.load_model('./assets/number_model.h5')

    # Probability of all numbers
    predictions = model.predict(image)
    top_3_idx = np.argsort(predictions[0])[-3:]

    # print("Probability", predictions[0])
    for idx, value in enumerate(top_3_idx[::-1]):
        percent = round(predictions[0][value] * 100, 2)
        print(f"{idx + 1}. Predict {value} with probability = {percent}%")

    # Prediction
    text = f"Prediction = {np.argmax(predictions[0])}"
    status.configure(text=text, fg="deepskyblue")
    print(f"\n{text}")


if __name__ == '__main__':
    ROOT = tk.Tk()
    ROOT.title("Number Guesser")
    ROOT.resizable(False, False)
    ROOT.configure(background=DKBLUE)
    ROOT.bind('<Escape>', lambda e: ROOT.quit())

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    status = Label(
        ROOT, text="Please draw a number!", 
        padx=10, font=20, fg="deepskyblue", bg=DKBLUE
    )
    status.pack(pady=20)

    # First Frame
    draw_frame = Frame(ROOT, pady=10, bg=DKBLUE)
    draw_frame.pack()

    canvas = Canvas(draw_frame, bg=DARK, width=500, height=500)
    canvas.pack()

    # Second Frame
    input_frame = Frame(ROOT, pady=10, bg=DKBLUE)
    input_frame.pack()

    # button
    predict_button = HooverButton(input_frame, text="Predict", command=predict)
    predict_button.grid(row=0, column=0, padx=10, sticky='news')

    save_button = HooverButton(input_frame, text="Save", command=save_image)
    save_button.grid(row=0, column=1, padx=10, sticky='news')

    erase_button = HooverButton(input_frame, text="Clear", command=erase_canvas)
    erase_button.grid(row=0, column=2, padx=10, sticky='news')

    # Left Click to draw
    canvas.bind("<B1-Motion>", draw_line)

    # Right CLick to erase
    canvas.bind("<B3-Motion>", erase_line)

    ROOT.mainloop()