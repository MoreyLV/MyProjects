import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

print("Operation:")
print("1. Education")
print("2. Recognition")
operation = int(input())

if operation == 1:
    train_data = np.array([
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]).flatten(),
        np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ]).flatten(),
        np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],   #11-8
            [1, 1, 1]
        ]).flatten()
    ])
    train_data_size = train_data.shape[1]
    exp_data = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] #11
    ])
    neuron = 10

    weight = np.random.randn(train_data_size, neuron)
    bias = np.random.randn(neuron)

    ages = 1000000
    current = 1

    for epoch in range(ages):
        if current != ages:
            percent = (current / ages) * 100
            print(str(current) + " --- " + str(f"{percent:.2f}%"))
            current += 1

        first_layer = train_data.dot(weight) + bias

        sigmoid_output = sigmoid(first_layer)
        sigmoid_proiz = sigmoid_derivative(sigmoid_output)

        learning_rate = 0.01

        error = sigmoid_output - exp_data
        delta = error * sigmoid_proiz

        weight -= learning_rate * train_data.T.dot(delta)
        bias -= learning_rate * np.sum(delta, axis=0)

    final_output = sigmoid(train_data.dot(weight) + bias)

    np.savetxt("weights.txt", weight)
    np.savetxt("bias.txt", bias)

    result = np.where(final_output < 0.89, 0, 1)

    print(result)
    print("RAW")
    print(final_output)
else:
    recognition_data = input("Enter massive 3x5 of number image")
    recognition_data_massive = np.array(eval(recognition_data))
    recognition_data_massive_flaten = recognition_data_massive.flatten().reshape(1, -1)

    weight = np.loadtxt("weights.txt")
    bias = np.loadtxt("bias.txt")

    final_output = sigmoid(recognition_data_massive_flaten.dot(weight)+bias)
    indexated_final = np.argmax(final_output)

    print(indexated_final)
