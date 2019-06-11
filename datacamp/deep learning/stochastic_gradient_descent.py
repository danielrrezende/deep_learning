## Specify Model - reads de data
pred = np.loadtxt('file.csv', delimiter=',')
input_shape = preds.shape[1]												# find the number of nodes in imput layer - number of columns = number os nodes in imput layer


#Funcion float range
def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step


def get_new_model(input_shape = input_shape):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_shape = input_shape))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	return(model)

lr_to_test = [.000001, 0.01, 1]
# lr_to_test = frange(10, 0, 0.1)

# loop over learning rates
for lr in lr_to_test:
	model = get_new_model()
	my_optimizer = SGD(lr=lr)
	model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
	early_stopping_monitor = EarlyStopping(patience=2)
	model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor])

