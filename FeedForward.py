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


class Network():
    def __init__(self,
        input_vector,
        True_Label,
        label_vector,
        alpha,#learning rate for backpropagation
        epochs,
        seed
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

        #set a seed for numpy generation to get reproducable results from random weight and bias generation
        np.random.seed(seed)

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
        for x in OBJ.network["hidden_layers"]:
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
        
    def backpropagation(self,i):
        DL=self.network["output_layer"]["activated_state"]-self.True_Label[i]
        for x in reversed(range(1,len(self.network["hidden_layers"])+1)):
            if x==len(self.network["hidden_layers"]):
                DW=np.dot(
                    np.multiply(
                        self.network["output_layer"]["weights"].T,
                        DL
                    ),
                    self.Softmax(self.network["output_layer"]["z_state"],True)
                )
                print(DW.shape)
                exit()

                DB=np.sum(DW.T,axis=1,keepdims=True)

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
                self.backpropagation(i)

            Loss=-np.sum(self.True_Label[i] * np.log(self.network["output_layer"]["activated_state"]))
            plt.plot(_,Loss,'go')
            print(f'Loss: {Loss} on epoch {_}')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gradient descent direction")
        plt.show()


data=np.array(pd.read_csv("train.csv"))
np.random.shuffle(data)
m,n=data.shape
#m=number of examples
#n=number of labels
data.T
#here we want to take 80% of the dataset in order to train it and 20% to validate
#however this is computationally demanding for a first pass 
#so we will be taking 32 images for training or 0.076% of the data
Y_train=np.array([int(data[x][:1]) for x in range(round(len(data)*.00076))])
X_train=np.array([data[x][1:]/n-1 for x in range(round(len(data)*.00076))])


#pull the next image from the randomized dataset
#a batch size of 33 will be 0.078%
Y_test=np.array([int(data[x][:1]) for x in range(round(len(data)*.00078))])[-1]
X_test=np.array([data[x][1:]/n-1 for x in range(round(len(data)*.00078))])[-1]
OBJ = Network(
    X_train,#normalized image data
    Y_train,#true label vector
    [0,1,2,3,4,5,6,7,8,9],#label vector for one-hot comparison
    0.0037,#learning rate
    40,#epochs
    11#seed for numpy random
)
OBJ.initialize_layers(84,2,["ReLU","Tanh"])
OBJ.Train()
results=OBJ.generate_report()
for x in results:
    print(x)
print(f"Network prediction {OBJ.get_prediction(X_test)}")
print(f"Real label: {Y_test}")
d=np.delete(data[32],0).reshape(28,28)
plt.imshow(d,cmap='gray', vmin=0, vmax=255)
plt.show()