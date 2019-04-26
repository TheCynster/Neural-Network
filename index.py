import numpy
import scipy.special

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        self.correct = 0.0
        pass
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        if numpy.argmax(targets) != numpy.argmax(final_outputs):
            output_errors = targets - final_outputs
            hidden_errors = numpy.dot(self.who.T, output_errors)
            self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        else:
            self.correct += 1.0
        return final_outputs
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs  = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs



training_data_file = open("mnist/mnist_test.csv","r")
training_data = training_data_file.read().split()
training_data_file.close()
for d in range(len(training_data)-1):
    training_data[d] = training_data[d].split(",")

n = neuralNetwork(784,100,10,0.0001)

for s in range(0,10):
    for t in range(len(training_data)-2):
        inputs = numpy.asfarray(training_data[t][1:])
        targets = numpy.asfarray([0]*10)#([training_data[t][0]])
        targets[int(training_data[t][0])] = 0.99
        n.train(inputs, targets)

