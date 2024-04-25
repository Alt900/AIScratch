import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#for fancy matplotlib effects
plt.rcParams['grid.color']="green"
plt.rcParams['axes.facecolor']="black"
plt.rcParams['figure.facecolor']="black"
plt.rcParams['text.color']="green"
plt.rcParams['axes.labelcolor']="green"
plt.rcParams['xtick.color']="green"
plt.rcParams['ytick.color']="green"

class Tools():
    def __init__(self):
        self.lossplot=plt
        self.lossplot.xlabel("Epoch")
        self.lossplot.ylabel("Loss")

    def plot_loss(self,L,_,color='g'):
        if _==0:
            self.lossplot.plot(_,L)
        else:
            self.lossplot.plot([_-1,_],[self.previousloss,L],color)
        self.previousloss=L

    def generate_report(self,network):
        r=[]
        for x in network["hidden_layers"]:
            r.append(f"internal hyperparameter shapes for hidden layer {x}:\nweights: {network['hidden_layers'][x]['weights'].shape}\nbias: {network['hidden_layers'][x]['bias'].shape}")
        return r

class Network():
    def __init__(self,
        input_vector,
        True_Label,
        label_vector,
        alpha,#learning rate for backpropagation
        epochs,
        seed,
        backpropagation_method,
        epsilon=None,
        rho=None,
        stoploss=0
    ):
        #input processing
        self.input_vector=input_vector
        #one-hot encode the true value labels and the labels
        encoded=np.zeros((len(label_vector),int(max(label_vector))+1))
        encoded[np.arange(len(label_vector)),label_vector]=1
        self.label_vector=encoded
        encoded=np.zeros((len(True_Label),int(max(True_Label))+1))
        encoded[np.arange(len(True_Label)),True_Label]=1
        self.True_Label=encoded

        #hyperparamaters
        self.alpha=alpha
        self.epochs=epochs
        self.epsilon=epsilon
        self.rho=rho
        self.stoploss=stoploss

        self.tool=Tools()

        #set a seed for numpy generation to get reproducable results from random weight and bias generation
        np.random.seed(seed)

        #backpropagation type
        self.backpropagation_method=backpropagation_method.lower()

        self.initialization={
            "Xavier":lambda i,o: np.random.uniform(low=-np.sqrt(6/(i+o)),high=np.sqrt(6/(i+o)),size=(o,i)),
            "He":lambda i,o: np.random.uniform(low=-np.sqrt(6/i),high=np.sqrt(6/i),size=(o,i))
        }

        self.Activations={
            "Tanh": lambda x,derivative :
                1-np.tanh(x)**2 if derivative \
                    else np.tanh(x)
            ,

            "ReLU": lambda zstate,derivative : np.array(
                [0 if x<0 else 1 for x in zstate] if derivative \
                    else [0 if x<0 else x for x in zstate]
            ),

            "Sigmoid": lambda x, derivative : 
                1/1+np.exp(x)*(1-(1/1+np.exp(x))) if derivative \
                else 1/1+np.exp(-x)
        }

        self.preinitdict={
            "function":None,#pointer to a lambda function chosen from the activation functions map
            "neurons":0,#int
            "weights":None,#2D matrix
            "bias":None,#1D vector
            "z_state":None,#1D array
            "activated_state":None#1D array
        }

        self.network={
            "input_layer":self.input_vector,
            "output_layer":dict(self.preinitdict),
            "hidden_layers":{},
            "LossFunction":None
        }
        self.network["output_layer"]["neurons"]=len(label_vector)




    def generate_report(self):
        r=[]
        for x in self.network["hidden_layers"]:
            r.append(f"internal hyperparameter shapes for hidden layer {x}:\nweights: {self.network['hidden_layers'][x]['weights'].shape}\nbias: {self.network['hidden_layers'][x]['bias'].shape}")
        return r

    def Softmax(self,vector,derivative=False):
        if derivative:
            n=len(vector)
            SM=self.Softmax(vector,derivative=False)
            JM=np.empty((n,n),dtype=type(SM))
            for i in range(n):
                for j in range(n):
                    if i==j:
                        JM[i,j]=SM[i]*(1-SM[j])
                    else:
                        JM[i,j]=-SM[i]*SM[j]
            return JM
        else:
            stable_exponential=np.exp(vector-np.max(vector))#prevent infinite overflow and underflow
            return stable_exponential/np.sum(stable_exponential)
        
    def backpropagation(self,i,epoch):
        DL=self.network["output_layer"]["activated_state"]-self.True_Label[i]
        for x in reversed(range(1,len(self.network["hidden_layers"])+1)):
            if x==len(self.network["hidden_layers"]):
                DW=np.dot(
                    np.multiply(
                        self.network["output_layer"]["weights"].T,
                        DL
                    ),
                    self.Softmax(self.network["output_layer"]["z_state"],True)
                ).astype(np.float64)

                DB=np.sum(DW.T,axis=1,keepdims=True).astype(np.float64)


                if self.backpropagation_method=="rmsprop":
                    if epoch==0:
                        self.ACCSG_W=self.rho*0+(1-self.rho)*(sum(sum(DW**2)))
                        self.ACCSG_B=self.rho*0+(1-self.rho)*(sum(DB**2))
                    else:
                        self.ACCSG_W=self.rho*self.ACCSG_W+(1-self.rho)*(sum(sum(DW**2)))
                        self.ACCSG_B=self.rho*self.ACCSG_B+(1-self.rho)*(sum(DB**2))

                    self.network["output_layer"]["weights"]=(self.network["output_layer"]["weights"].T-self.alpha*DW/(np.sqrt(self.ACCSG_W)+self.epsilon)).T
                    self.network["output_layer"]["bias"]=(self.network["output_layer"]["bias"]-self.alpha)*DB/(np.sqrt(self.ACCSG_B)+self.epsilon)

                else:
                    self.network["output_layer"]["weights"]=(self.network["output_layer"]["weights"].T-self.alpha*DW).T
                    self.network["output_layer"]["bias"]=self.network["output_layer"]["bias"]-self.alpha*DB


                DA=np.multiply(
                    self.network["output_layer"]["weights"].T.dot(DW.T),
                    self.network["hidden_layers"][x]["function"](
                        self.network["hidden_layers"][x]["z_state"],True
                    )
                )

                DW=DA.T.dot(self.network["hidden_layers"][x]["activated_state"])
                DB=DW

                if self.backpropagation_method=="rmsprop":
                    self.ACCSG_W=self.rho*self.ACCSG_W+(1-self.rho)*(sum(DW**2))
                    self.ACCSG_B=self.rho*self.ACCSG_B+(1-self.rho)*(sum(DB**2))

                    self.network["hidden_layers"][x]["weights"]=(self.network["hidden_layers"][x]["weights"].T-self.alpha*DW/(np.sqrt(self.ACCSG_W)+self.epsilon)).T
                    self.network["hidden_layers"][x]["bias"]=(self.network["hidden_layers"][x]["bias"].T-self.alpha*DB/(np.sqrt(self.ACCSG_B)+self.epsilon)).reshape(len(DB),1)

                else:
                    self.network["hidden_layers"][x]["weights"]=(self.network["hidden_layers"][x]["weights"].T-self.alpha*DW).T
                    self.network["hidden_layers"][x]["bias"]=(self.network["hidden_layers"][x]["bias"].T-self.alpha*DB).reshape(len(DB),1)

            elif x!=len(self.network["hidden_layers"]) and x!=1:
                DW=np.multiply(
                    self.network["hidden_layers"][x+1]["weights"].T.dot(DW),
                    self.network["hidden_layers"][x]["function"](
                        self.network["hidden_layers"][x]["z_state"],True
                    )
                )

                DW=DW*(self.network["hidden_layers"][x]["activated_state"])
                DB=DW

                if self.backpropagation_method=="rmsprop":
                    self.ACCSG_W=self.rho*self.ACCSG_W+(1-self.rho)*(sum(DW**2))
                    self.ACCSG_B=self.rho*self.ACCSG_B+(1-self.rho)*(sum(DB**2))

                    self.network["hidden_layers"][x]["weights"]=((self.network["hidden_layers"][x]["weights"].T-self.alpha)*DW/(np.sqrt(self.ACCSG_W)+self.epsilon)).T
                    self.network["hidden_layers"][x]["bias"]=(self.network["hidden_layers"][x]["bias"].T-self.alpha*DB/(np.sqrt(self.ACCSG_B)+self.epsilon)).reshape(len(DB),1)

                else:
                    self.network["hidden_layers"][x]["weights"]=(self.network["hidden_layers"][x]["weights"].T-self.alpha*DW).T
                    self.network["hidden_layers"][x]["bias"]=(self.network["hidden_layers"][x]["bias"].T-self.alpha*DB).reshape(len(DB),1)

            else:
                pass

    def get_prediction(self,image):
        for x in range(1,self.layercount):
            z=np.empty(shape=self.network["hidden_layers"][x]["neurons"])
            for y in range(self.network["hidden_layers"][x]["neurons"]):
                if x==1:
                    z.flat[y]=np.sum(np.dot(
                        self.network["hidden_layers"][x]["weights"][y],
                        image
                    )+self.network["hidden_layers"][x]["bias"][y])
                else:
                    z.flat[y]=np.sum(np.dot(
                        self.network["hidden_layers"][x]["weights"][y],
                        self.network["hidden_layers"][x-1]["activated_state"]
                    )+self.network["hidden_layers"][x]["bias"][y])
            self.network["hidden_layers"][x]["z_state"]=z
            self.network["hidden_layers"][x]["activated_state"]=self.network["hidden_layers"][x]["function"](z,False)

        z=np.empty(shape=self.network["output_layer"]["neurons"])
        for y in range(self.network["output_layer"]["neurons"]):
            z.flat[y]=np.sum(np.multiply(
                self.network["output_layer"]["weights"][y],
                self.network["hidden_layers"][x]["activated_state"]
            )+self.network["output_layer"]["bias"][y])
        self.network["output_layer"]["z_state"]=z
        self.network["output_layer"]["activated_state"]=self.Softmax(z)
        z=np.empty(shape=self.network["output_layer"]["neurons"])
        for y in range(self.network["output_layer"]["neurons"]):
            z.flat[y]=np.sum(np.multiply(
                self.network["output_layer"]["weights"][y],
                self.network["hidden_layers"][x]["activated_state"]
            )+self.network["output_layer"]["bias"][y])
        self.network["output_layer"]["z_state"]=z
        self.network["output_layer"]["activated_state"]=self.Softmax(z)
        print(self.network["output_layer"]["activated_state"])
        predictedlabel=np.where(self.network["output_layer"]["activated_state"]==max(self.network["output_layer"]["activated_state"]))[0][0]
        return predictedlabel

    def initialize_layers(self,neurondist,layercount,activation):
        self.layercount=layercount+1
        if type(neurondist)==int:
            neurondist=[neurondist//y for y in reversed([x for x in range(1,self.layercount)])]
        if type(activation)==str:
            if activation=="Softmax":
                print("Cannot add softmax activation for hidden layer activation functions.")
                exit()
            if type(activation)==list and len(activation)!=self.layercount:
                print("The list of activation functions are not complete")
                exit()
            else:
                activation=[activation for y in range(1,self.layercount)]
        for x in range(1,self.layercount):
            self.network["hidden_layers"][x]=dict(self.preinitdict)#shallow copy 
            self.network["hidden_layers"][x]["neurons"]=neurondist[x-1]
            args=[[neurondist[x-1],len(self.input_vector[0])] if x==1 else [neurondist[x-1],neurondist[x-2]]]
            self.network["hidden_layers"][x]["function"]=self.Activations[activation[x-1]]
            if activation[x-1]=="ReLU" or activation[x-1]=="Leaky ReLU":
                self.network["hidden_layers"][x]["weights"]=self.initialization["He"](args[0][1],args[0][0])
                self.network["hidden_layers"][x]["bias"]=self.initialization["He"](1,args[0][0])
            else:
                self.network["hidden_layers"][x]["weights"]=self.initialization["Xavier"](args[0][1],args[0][0])
                self.network["hidden_layers"][x]["bias"]=self.initialization["Xavier"](1,args[0][0])
        self.network["output_layer"]["weights"]=self.initialization["Xavier"](args[0][0],len(self.label_vector))
        self.network["output_layer"]["bias"]=self.initialization["Xavier"](1,len(self.label_vector))
            

    def Train(self):
        for _ in range(0,self.epochs):
            for i in range(len(self.input_vector)):
                for x in range(1,self.layercount):
                    z=np.empty(shape=self.network["hidden_layers"][x]["neurons"])
                    for y in range(self.network["hidden_layers"][x]["neurons"]):
                        if x==1:#weights shape (20,784), input vector shape = 784
                            z.flat[y]=np.sum(np.dot(
                                self.network["hidden_layers"][x]["weights"][y],
                                self.input_vector[i]
                            )+self.network["hidden_layers"][x]["bias"][y])
                        else:
                            z.flat[y]=np.sum(np.dot(
                                self.network["hidden_layers"][x]["weights"][y],
                                self.network["hidden_layers"][x-1]["activated_state"]
                            )+self.network["hidden_layers"][x]["bias"][y])
                    self.network["hidden_layers"][x]["z_state"]=z
                    self.network["hidden_layers"][x]["activated_state"]=self.network["hidden_layers"][x]["function"](z,False)

                z=np.empty(shape=self.network["output_layer"]["neurons"])
                for y in range(self.network["output_layer"]["neurons"]):
                    z.flat[y]=np.sum(np.multiply(
                        self.network["output_layer"]["weights"][y],
                        self.network["hidden_layers"][x]["activated_state"]
                    )+self.network["output_layer"]["bias"][y])
                self.network["output_layer"]["z_state"]=z
                self.network["output_layer"]["activated_state"]=self.Softmax(z)
                self.backpropagation(i,_)

            Loss=-np.sum(self.True_Label[i] * np.log(self.network["output_layer"]["activated_state"]))
            if Loss<self.stoploss:
                print(f"Stoploss met at epoch {_} on a loss of {Loss}")
                self.tool.plot_loss(Loss,_,'r')
                break

            print(f"Loss {Loss} at epoch {_}")
            self.tool.plot_loss(Loss,_)
        self.tool.lossplot.title(f"{self.backpropagation_method} direction")
        self.tool.lossplot.show()
