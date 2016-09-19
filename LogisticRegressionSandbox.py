# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 18:01:09 2016

@author: alana
"""
import random as rand
import numpy as np
import math

#simple class to hold x1 and x2 parameters of the input
class DP:
    y= 0 # just establshing that it exists... habit from C++ really, but I'll do it anyway in case
    x0=1 #variable just to be able to be dot-producted with the w0 of the function
    def __init__ (self, _x1,_x2):
        self.x1=_x1
        self.x2=_x2
                
        
class Function:
    def __init__(self,_w0,_w1,_w2):
        self.w0=_w0
        self.w1=_w1
        self.w2=_w2
        
#evaluates a point given a test function and a data point
def EvaluatePoint(dp, func):
    value = func.w0+(dp.x1*func.w1)+(dp.x2*func.w2)
    #returns the sign of w^T*x
    if (value>=0):
        return (1)
    else:
        return (-1)
        

#generate nPoints data points
def GenerateTestData(nPoints, targetFunction):
    dataSet = []
    for i in range(0,nPoints):
        x = DP(rand.uniform(-1.0,1.0),rand.uniform(-1.0,1.0))
        x.y = EvaluatePoint(x,targetFunction)
        dataSet.append(x)
    return dataSet
        
#applies logistic growth function to the signal w^t*x
#for this case, we simply apply tanch as the growth function
def Theta(s):
    theta = math.exp(s)/(1+math.exp(s))
    return theta
        
        
#returns a value for the growht function that we call theta
def CalculateEIn(_dataSet, _hypothesis):
    
    totalError = 0
    for i in range (0,len(_dataSet)-1):
        x_n = _dataSet[i]
        #boils down to ln(1+e^(-y*(x_n dot w^T)))
        error=np.log(1+math.exp(-1*x_n.y*((_hypothesis.w0*x_n.x0)+(_hypothesis.w1*x_n.x1)+(_hypothesis.w2*x_n.x2))))
        totalError+=error
    
    totalError/=len(_dataSet) #takes average
    #print(str(totalError))
    return totalError
        


#stochastic Gradient Descent Function
def StochasticGD(_dataSet, _alpha, _numIterations):
    
    hypo = Function(0,0,0)
    bestW0=0.0
    bestW1=0.0
    bestW2=0.0
    
    eIn = CalculateEIn(_dataSet, hypo) #initial value of in-sample error
    
    for i in range(0,_numIterations):
        x_n = _dataSet[rand.randint(0,nPoints-1)] #stochastically choose a point
        #this is the gradient, without considering parameter x yet... The weight update will account for that
        gradient=-1*x_n.y/(1+(math.exp(x_n.y*((hypo.w0*x_n.x0)+(hypo.w1*x_n.x1) + (hypo.w2*x_n.x2)))))
        #only update the weights if the new in-sample error was lower than the previous
        #update weights now
        hypo.w0 = hypo.w0 - (_alpha*gradient*x_n.x0)
        hypo.w1 = hypo.w1 - (_alpha*gradient*x_n.x1)
        hypo.w2 = hypo.w2 - (_alpha*gradient*x_n.x2)
        
        #calculate new E_in
        newEIn=CalculateEIn(_dataSet,hypo)

        #if the new E_in is lower than the old E_in, "pocket" the values
        if (newEIn<=eIn):
            bestW0=hypo.w0
            bestW1=hypo.w1
            bestW2=hypo.w2
          
    _g = Function(bestW0,bestW1,bestW2)
    return _g

#some necessary variables
nPoints = 100
alpha = .1 #learning rate
numIterations = 1000 #simply the number of iterations of stochastic GD        
        
#def GradientDescent:     
targetF = Function(rand.uniform(-1.0,1.0),rand.uniform(-1.0,1.0),rand.uniform(-1.0,.0))
dataSet = GenerateTestData(nPoints,targetF)
g = StochasticGD(dataSet, alpha, numIterations)
for i in range(0,len(dataSet)-1):
    if (dataSet[i].y==EvaluatePoint(dataSet[i],g)):
        print("true")
    else:
        print("false")
print ("TargetFunction Weights:\nw0:"+str(targetF.w0) + ";\nw1: " +str(targetF.w1) + ";\nw2: " + str(targetF.w2))
print ("Hypothesis Weights:\nw0:"+str(g.w0) + ";\nw1: " +str(g.w1) + ";\nw2: " + str(g.w2))
print ("\nE_in is: " + str(CalculateEIn(dataSet,g)))
testData = GenerateTestData(nPoints, targetF)
eOut = CalculateEIn(testData, g)
print ("E_out is: " + str(eOut))