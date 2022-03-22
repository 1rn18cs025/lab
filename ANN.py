import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0)
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))

epoch=5000
lr=0.1
inputlayer_neurons = 2 		
hiddenlayer_neurons = 3 	
output_neurons = 1 		

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

    hinp=np.dot(X,wh) + bh
    hlayer_act = sigmoid(hinp)
    outinp= np.dot(hlayer_act,wout)+ bout
    output = sigmoid(outinp)

    EO = y-output
    d_output = EO*output*(1-output)
    EH = np.dot(wout.T,d_output)

    d_hiddenlayer = EH * hlayer_act*(1-hlayer_act)


    wout += np.dot(hlayer_act.T,d_output) *lr
    wh +=np.dot(X.T,d_hiddenlayer ) *lr

print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
