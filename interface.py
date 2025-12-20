import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np

class DrawApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Draw a Digit")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        tk.Button(self.root, text="Predict", command=self.predict).pack()
        tk.Button(self.root, text="Clear", command=self.clear).pack()

        self.root.mainloop()

    def paint(self, event):
        brush_size = 10 # smaller
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)

        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)

    
    def preprocess(self, img):
        img = img.convert('L')
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            # Add a small padding to avoid squashing zeros
            pad = 2
            left, upper, right, lower = bbox
            bbox = (max(left-pad,0), max(upper-pad,0), min(right+pad,img.width), min(lower+pad,img.height))
            img = img.crop(bbox)

        # Resize while keeping aspect ratio
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)

        new_img = Image.new('L', (28, 28), 0)
        left = (28 - img.width) // 2
        top = (28 - img.height) // 2
        new_img.paste(img, (left, top))

        img_array = np.array(new_img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)


        return 1-img_tensor

        

    def predict(self):


        img_tensor = self.preprocess(self.image)
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(1).item()
        
        img_tensor = self.preprocess(self.image)

        #img_to_check = Image.fromarray((img_tensor.squeeze().numpy() * 255).astype(np.uint8))
        #img_to_check.save("debug_input.png")

        messagebox.showinfo("Prediction", f"The model predicts: {pred}")
