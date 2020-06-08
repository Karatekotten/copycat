import numpy as np
import cv2

from pynput.keyboard import Listener, Controller, Key, KeyCode
from PIL import ImageGrab
from time import sleep
from collections import deque

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout

frame_rate = 15
frame_max = 2
frame_size = (128, 72)
capture_box = (419, 151, 1699, 871)
passivity = False

directory = '.keras/datasets/'
filename_x = 'capture_list_x'
filename_y = 'capture_list_y'
filename_model = 'screen_capture.model'

input_check = [Key.down, Key.up, Key.left, Key.right, 'z', 'x']

def record():
	global RECORDING
	RECORDING = True

	print('Recording Started, Press Esc To Stop')

	#input_check = [Key.down, Key.up, Key.left, Key.right, 'z', 'x']
	input_state = np.zeros(len(input_check))

	# Keyboard Input
	def key_press(key):
		if type(key) == KeyCode:
			key = key.char

		if key in input_check:
			input_state[input_check.index(key)] = True

	def key_release(key):
		if type(key) == KeyCode:
			key = key.char

		if key in input_check:
			input_state[input_check.index(key)] = False

		if key == Key.esc:
			global RECORDING
			RECORDING = False

	listener = Listener(on_press=key_press, on_release=key_release)
	listener.start()

	# Capture Image Sequence
	video_list = []
	input_list = []

	video = deque([])

	while True:
		capture = ImageGrab.grab(bbox=capture_box)
		img = cv2.cvtColor(np.array(capture), cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, frame_size)

		video.append(img)

		# Add Sample To List
		if len(video) >= frame_max:
			if not np.all(input_state==0) or passivity == True:
				video_list.append(video.copy())
				input_list.append(input_state.copy())
				print('Samples:', len(input_list), end='\r')

			video.clear()
		
		# Save Dataset And Stop Recording
		if (RECORDING == False):
			np.save(directory+filename_x, video_list)
			np.save(directory+filename_y, input_list)
			
			print('Saved Dataset ', directory + filename_x + '.npy, ' + directory + filename_y + '.npy')

			break

		sleep(1/frame_rate)

	listener.stop()

	print('Recording Stopped')

def train():
	x = np.load(directory+filename_x+'.npy')/255
	y = np.load(directory+filename_y+'.npy')

	# Add Number Of Color Channels
	x = np.expand_dims(x, axis=-1)

	input_shape = x.shape[1:]
	output_shape = y.shape[-1]

	# Create Model
	model = Sequential()

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=input_shape, padding='same', return_sequences=True))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Flatten())
	model.add(Dense(output_shape, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Train Model
	model.fit(x, y, epochs=100, batch_size=10)

	# Save Model
	model.save(filename_model)

def simulate():
	global SIMULATING
	SIMULATING = True

	print('Simulate Started, Press Esc To Stop')

	# Load Model
	model = load_model(filename_model)

	# Keyboard
	#input_check = [Key.down, Key.up, Key.left, Key.right, 'z', 'x']
	input_state = np.zeros(len(input_check))
	input_state_previous = input_state.copy()

	def key_release(key):
		if key == Key.esc:
			global SIMULATING
			SIMULATING = False

	listener = Listener(on_release=key_release)
	listener.start()

	controller = Controller()

	# Capture Image Sequence
	video = np.empty((0, frame_size[1], frame_size[0]), dtype=np.float)

	while True:
		capture = ImageGrab.grab(bbox=capture_box)
		img = cv2.cvtColor(np.array(capture), cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, frame_size)

		video = np.concatenate((video, img[None, ::, ::]), axis=0)

		if len(video) == frame_max:
			x = np.expand_dims(video/255, axis=-1)
			y = model.predict(np.expand_dims(x, axis=0))

			y[y < 0.5] = False
			y[y > 0.5] = True

			input_state = y[0]

			print(input_state)
			
			# Simulate Output
			for i in range(len(input_state)):
				#if (input_state[i] == True) and (input_state_previous[i] == False):
				if (input_state[i] == True):
					print('press ', input_check[i])
					controller.press(input_check[i])

				if (input_state[i] == False) and (input_state_previous[i] == True):
					print('release ', input_check[i])
					controller.release(input_check[i])
			
			input_state_previous = input_state.copy()

			video = np.delete(video, 0, 0)

		if (SIMULATING == False):
			break
		
		sleep(1/frame_rate)

	listener.stop()

	print('Simulate Stopped')