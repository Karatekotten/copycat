import copycat

while True:
	print('----------------MENU---------------')
	print('| 1. record dataset                ')
	print('| 2. train model                   ')
	print('| 3. run                           ')
	print('| 4. get input                     ')
	print('| 5. set input                     ')
	print('| 0. exit                          ')
	print('-----------------------------------')

	command = input('input:')

	if command == '1':
		copycat.record()
	elif command == '2':
		copycat.train()
	elif command == '3':
		copycat.run()
	elif command == '4':
		copycat.get_input()
	elif command == '5':
		copycat.set_input()
	elif command == '0':
		break
	else:
		print('Unkown Command')