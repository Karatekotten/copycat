import numpy as np
import cv2

import cinput

from pynput.keyboard import Listener, Controller, Key, KeyCode
from PIL import ImageGrab
from time import sleep
from collections import deque
from pathlib import Path
from configparser import ConfigParser

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout, MaxPooling3D, MaxPooling2D

directory = Path(__file__).parent.absolute()

config = ConfigParser()
config.read(directory.joinpath('settings.ini'))

frame_rate = config.getint('RECORD', 'frame_rate')
frame_max = config.getint('RECORD', 'frame_max')
frame_width = config.getint('RECORD', 'frame_width')
frame_height = config.getint('RECORD', 'frame_height')

passivity = config.getboolean('RECORD', 'passivity')

capture_x = config.getint('RECORD', 'capture_x')
capture_y = config.getint('RECORD', 'capture_y')
capture_w = config.getint('RECORD', 'capture_w')
capture_h = config.getint('RECORD', 'capture_h')

capture_box = (capture_x, capture_y, capture_x+capture_w, capture_y+capture_h)

filename_x = config.get('FILES', 'filename_x')
filename_y = config.get('FILES', 'filename_y')
filename_model = config.get('FILES', 'filename_model')

input_check = [Key.down, Key.up, Key.left, Key.right]

def record():
	global RECORDING
	RECORDING = True

	print('Recording Started, Press Esc To Stop')

	# Keyboard Input
	input_state = np.zeros(len(input_check))

	def key_press(key):
		if (type(key) == KeyCode):
			key = key.char

		if (key in input_check):
			input_state[input_check.index(key)] = True

	def key_release(key):
		if (type(key) == KeyCode):
			key = key.char

		if (key in input_check):
			input_state[input_check.index(key)] = False

		if (key == Key.esc):
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
		img = cv2.resize(img, (frame_width, frame_height))

		video.append(img)

		# Add Sample To List
		if (len(video) >= frame_max):
			if not np.all(input_state==0) or (passivity == True):
				video_list.append(video.copy())
				input_list.append(input_state.copy())

				print('Samples:', len(input_list), end='\r')

			video.clear()
		
		# Save Dataset And Stop Recording
		if (RECORDING == False):
			np.save(directory.joinpath('datasets').joinpath(filename_x), video_list)
			np.save(directory.joinpath('datasets').joinpath(filename_y), input_list)
			
			print('Dataset Saved', len(input_list), 'Samples')

			break

		sleep(1/frame_rate)

	listener.stop()

	print('Recording Stopped')

def train():
	# Load Dataset
	x = np.load(directory.joinpath('datasets').joinpath(filename_x+'.npy'))/255
	y = np.load(directory.joinpath('datasets').joinpath(filename_y+'.npy'))

	# Add Number Of Color Channels
	x = np.expand_dims(x, axis=-1)

	input_shape = x.shape[1:]
	output_shape = y.shape[-1]

	# Create Model
	model = Sequential()

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=input_shape, padding='same', return_sequences=True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=False))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Flatten())

	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(output_shape, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

	# Train Model
	model.fit(x, y, epochs=200, batch_size=10, validation_split=0.05)
	
	# Save Model
	model.save(str(directory.joinpath('models').joinpath(filename_model)))

def run():
	global RUN
	RUN = True

	print('Running, Press Esc To Stop')

	# Load Model
	model = load_model(str(directory.joinpath('models').joinpath(filename_model)))

	# Keyboard
	input_state = np.zeros(len(input_check))
	input_state_previous = input_state.copy()

	def key_release(key):
		if (key == Key.esc):
			global RUN
			RUN = False

	listener = Listener(on_release=key_release)
	listener.start()

	# Capture Image Sequence
	video = np.empty((0, frame_height, frame_width), dtype=np.float)

	while True:
		# Capture Frame
		capture = ImageGrab.grab(bbox=capture_box)
		img = cv2.cvtColor(np.array(capture), cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (frame_width, frame_height))

		# Add To Sequence
		video = np.concatenate((video, img[None, ::, ::]), axis=0)

		if (len(video) == frame_max):
			# Predict Output
			x = np.expand_dims(video/255, axis=-1)
			y = model.predict(np.expand_dims(x, axis=0))

			y[y < 0.5] = False
			y[y > 0.5] = True

			input_state = y[0]

			# Simulate Output
			for i in range(len(input_state)):
				if (input_state[i] == True) and (input_state_previous[i] == False):
					cinput.PressKey(cinput.scancode[input_check[i]])

				if (input_state[i] == False) and (input_state_previous[i] == True):
					cinput.ReleaseKey(cinput.scancode[input_check[i]])
			
			input_state_previous = input_state.copy()

			# Remove Oldest Frame
			video = np.delete(video, 0, 0)

		# End Simulation
		if (RUN == False):
			for i in range(len(input_state)):
				if (input_state[i] == True):
					cinput.ReleaseKey(cinput.scancode[input_check[i]])
			
			break
		
		sleep(1/frame_rate)

	listener.stop()

	print('Stopped')

def get_input():
	print(input_check)

def set_input():
	print('Input Cleared, Press Keys To Add To Input, Press Esc To Save:')

	input_check.clear()

	def key_press(key):
		if (key == Key.esc):
			print('New Input:', input_check)
			return False
		
		if (type(key) == KeyCode):
			key = key.char

		if (key in input_check):
			print(key, 'already exists in input')
		else:
			input_check.append(key)
			print(key, 'added to input')

	with Listener(on_press=key_press) as listener:
		listener.join()