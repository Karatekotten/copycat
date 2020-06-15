import numpy as np
from pathlib import Path
import sys

directory = Path(__file__).parent.absolute()

x = np.load(directory.joinpath('datasets').joinpath('capture_list_x.npy'))
y = np.load(directory.joinpath('datasets').joinpath('capture_list_y.npy'))

print('size of x:', x.nbytes/1000, 'kb')
print('size of y:', y.nbytes/1000, 'kb')

samples = x.shape[0]
frames = x.shape[1]
frame_height = x.shape[2]
frame_width = x.shape[3]

import tkinter as tk
from PIL import ImageTk, Image

# Window
window_width = frame_width*(frames+1)
window_height = frame_height*samples

root = tk.Tk('Samples')
root.resizable(False, False)

box=tk.Frame(root, width=window_width, height=window_height)
box.pack(expand=True, fill=tk.Y)

# Canvas
canvas = tk.Canvas(box, width=window_width, height=960)

def scroll_wheel(e):
	if (e.delta < 0):
		canvas.yview_scroll(1, 'units')
	
	if (e.delta > 0):
		canvas.yview_scroll(-1, 'units')

root.bind("<MouseWheel>", scroll_wheel)

scrollbar = tk.Scrollbar(box)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar.config(command=canvas.yview)

canvas.pack(expand=0, fill=tk.BOTH)

# Draw Samples Scroll
image_list = []

for s in range(samples):
	for f in range(frames):
		image = ImageTk.PhotoImage(image=Image.fromarray(x[s][f]))
		image_list.append(image)
		canvas.create_image(f*frame_width, s*frame_height, anchor='nw', image=image_list[-1])

	canvas.create_text(frame_width*frames, s*frame_height, anchor='nw', text=s)
	canvas.create_text(frame_width*frames, s*frame_height, anchor='sw', text=y[s])

canvas.configure(scrollregion=canvas.bbox('all'))

root.mainloop()