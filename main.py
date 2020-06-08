import copycat

while True:
	print('----------------MENU---------------')
	print('| 1. rec                          |')
	print('| 2. train                        |')
	print('| 3. sim                          |')
	print('| 4. exit                         |')
	print('-----------------------------------')

	command = input('input:')

	if command == 'rec' or command == '1':
		copycat.record()
	elif command == 'train' or command == '2':
		copycat.train()
	elif command == 'sim' or command == '3':
		copycat.simulate()
	elif command == 'exit' or command == '4':
		break
	else:
		print('Unkown Command')