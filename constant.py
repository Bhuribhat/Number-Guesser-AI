from tkinter import *

# constants
RATIO  = 28
WIDTH  = 560
HEIGHT = 560

MARKER_SIZE = (WIDTH // RATIO)

# colors
GRAY   = "#545454"
DARK   = "#1F2937"
DKBLUE = "#111827"
DARKER = "#1F2937"
PURPLE = "#8B5CF6"
YELLOW = "#fece2f"
WHITE  = "#FFFFFF"


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


class CustomEntry(Entry):
    def __init__(self, *args, **kwargs):
        Entry.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["width"] = 30
        self["fg"] = YELLOW
        self["bg"] = DARKER
        self["insertbackground"] = "orange"

    def set(self, text):
        self.delete(0, END)
        self.insert(0, text)


class CustomLabel(Label):
    def __init__(self, *args, **kwargs):
        Label.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["fg"] = "white"
        self["bg"] = DKBLUE
        self["padx"] = 10
        self["pady"] = 5


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