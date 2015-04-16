'''
Created on 26-Mar-2015
@author: Anantharaman Narayana Iyer
Implementation of Softmax classifier with bias units instead of extra input 
'''
import random
import scipy.optimize
import numpy
import math

class Softmax(object):
    '''
    classdocs
    '''
    def __init__(self, input_size, num_classes, regularizer = 0.0001):
        '''
        Constructor
        '''
        self.input_size = input_size
        self.num_classes = num_classes
        self.reg = regularizer # this is our lambda
        #self.model = self.init_model(self.input_size + 1, self.num_classes)
        self.model = self.init_model(self.input_size, self.num_classes)
        #print len(self.model)
        self.offset_0 = 0
        #self.offset_1 = (self.input_size + 1) * self.num_classes # this marks the end of weights list
        self.offset_1 = (self.input_size) * self.num_classes # this marks the end of weights list
        self.offset_2 = self.offset_1 + self.num_classes # this includes the bias units
        self.iteration_count = 0
        return
    
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        #print "SETTING the model"
        self.model = model
        return
    
    def init_model(self, size, classes):
        # given the number of features (size) and output classes return an initial model
        model = []
        low = -0.0001
        hi = 0.001
        pop_size = 100000
        population = [random.uniform(low, hi) for i in range(pop_size)] # generate random numbers as a large population
        weights = random.sample(population, size * classes) # generate a list of random numbers that represent the model
        model.extend(weights)
        biases = [0] * classes
        model.extend(biases)
        return model
    
    def p_y_given_x(self, theta, biases, xi): 
        # theta is k x n matrix where k = num classes, n = num features
        # xi is n x 1
        val = numpy.dot(theta, xi) + biases
        #print "val = ", val, "max = ", max(val)
        val = numpy.subtract(val, max(val))
        #print "val = ", val, "max = ", max(val)
        numerator = numpy.exp(val)
        denominator = sum(numerator)
        prob = numerator / denominator
        
        for n in numerator:
            if n == 0:
                print "ERROR: prob term getting zero, val = ", val, " theta = ", theta, " num = ", numerator, " den = ", denominator, " p = ", prob
        
        return prob
    
    def compute_grads(self, probs, xi, yi): # given the probs and xi return grads
        #print "xi = ", xi, " yi = ", yi
        grads = []
        biases = []
        for k in range(self.num_classes): # compute a grad delta_j for each j from 1 to k
            val = yi[k] - probs[k] # this is a scalar value
            grads.append(numpy.multiply(val, xi))
            biases.append(val)
        grads = numpy.array(grads)
        biases = numpy.array(biases)
        return (grads, biases, )
    
    def compute_cost(self, theta_1, data, weight_decay):    
        #print "in cost func decay = ", weight_decay
        grads = []
        delta_biases = []
            
        #biases = theta_1[self.offset_1 : ]        
        # initialize gradients
        for i in range(self.num_classes):
            #val = [0.0] * (self.input_size + 1)
            val = [0.0] * (self.input_size )
            grads.append(val)        
        grads = numpy.array(grads)

        for i in range(self.num_classes):
            #val = [0.0] * (self.input_size + 1)
            val = 0.0
            delta_biases.append(val)        
        delta_biases = numpy.array(delta_biases)
            
        theta = theta_1[self.offset_0 : self.offset_1]    
        biases = theta_1[self.offset_1 :]    
        regularization = 0.5 * weight_decay * sum([t * t for t in theta])
        #W1 = theta.reshape(self.num_classes, self.input_size + 1)
        W1 = theta.reshape(self.num_classes, self.input_size )
        outer_sum = 0.0 # this is the cost across all training examples
        for i in range(len(data["X"])):
            xi = data["X"][i] # this is the ith training example
            yi = data["Y"][i] # this is a row vector of length k - 1 x k
            probs = self.p_y_given_x(W1, biases, numpy.transpose(xi))
            log_probs = numpy.log(probs) # this is a vector of 1 x k
            #print "p = ", probs, " lp = ", log_probs
            g, b = self.compute_grads(probs, xi, yi) # get the gradients for k classes
            grads = numpy.add(grads, g)
            delta_biases = numpy.add(delta_biases, b)
            outer_sum += sum(numpy.multiply(yi, log_probs))
        
        cost = ((-1.0 / len(data["X"])) * outer_sum) + regularization
        #print outer_sum, cost
        grads = (-1.0 / len(data["X"])) * grads        
        grad_regularization = numpy.multiply(weight_decay, W1)
        grads = numpy.add(grads, grad_regularization)

        delta_biases = (-1.0 / len(data["X"])) * delta_biases        

        '''
        if (self.iteration_count % 1) == 0:
            print 'in iteration: ', self.iteration_count
            print 'SM COST = ', cost #, "  grads shape = ", grads.shape, "  grads = ", grads, " w1 shape = ", W1.shape
            print "SM BIASES = ", biases
        '''

        self.iteration_count += 1
        
        deltas =  numpy.concatenate((grads.flatten(), delta_biases.flatten(),))
        #print "len of deltas = ", len(deltas), " len of grads = ", len(grads.flatten()), " len of biases = ", len(delta_biases.flatten())
        
        return [cost, deltas]
    
    def predict(self, xi): # given the input xi predict the prob distribution across output classes
        theta = self.model[self.offset_0 : self.offset_1]    
        biases = self.model[self.offset_1 : ]    
        #W1 = theta.reshape(self.num_classes, self.input_size + 1)
        W1 = theta.reshape(self.num_classes, self.input_size )
        #xi1 = [1.0]
        #xi1.extend(xi)
        probs = self.p_y_given_x(W1, biases, xi)
        return probs        
    
    def predict_all(self, X): # given the input xi predict the prob distribution across output classes
        #print "In SM, predict_all"
        theta = self.model[self.offset_0 : self.offset_1]    
        biases = self.model[self.offset_1 : ]    
        #W1 = theta.reshape(self.num_classes, self.input_size + 1)
        W1 = theta.reshape(self.num_classes, self.input_size )

        ret = []
        for xi in X:
            #xi1 = [1.0]
            #xi1.extend(xi)
            probs = self.p_y_given_x(W1, biases, xi)
            ret.append(probs)
        return ret        
    
    
    def train(self, training_data, weight_decay = 0.001, max_iterations = 1000): 
        # data is of the form: {"xmatrix": xmatrix, "ymatrix": yvec}
        # xmatrix will have m rows representing training examples and n columns where each column is a feature. 
        # ymatrix is a matrix of expected outputs across K classes. Hence ymatrix is: m x K
        # given the training data perform the training of Softmax
        #training_data["X"] = numpy.c_[numpy.ones(len(training_data["X"])), training_data["X"]]
        model  = scipy.optimize.minimize(self.compute_cost, self.model, 
                                            args = (training_data, weight_decay), method = 'L-BFGS-B', 
                                            jac = True) #  , options = {'maxiter': max_iterations}
        self.model = model.x
        return model

if __name__ == "__main__":
    # X = matrix of m input examples and n features
    # Y = matrix of m output vectors, where each output vector has k output units
    # create a dictionary: data = {"X": X, "Y": Y}
    # create a Softmax instance: sm = Softmax(num_of_features, num_of_output_units)
    # Train the classifier: sm.train(X, Y)
    # predict a single input by: probs = sm.predict(X[i])
    # probs is the probability distribution across k classes

    
