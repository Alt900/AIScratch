# Artificial Intelligence from scratch
The more prevalent artificial intelligence becomes in the modern world and especially with the rise of ChatGPT more people are drawn to artificial intelligence.
From my experience with AI going as far back as 2018, there were a lot of pitfalls back then for starting down the path of artificial intelligence and it was a very steep learning curve.
Especially now with ChatGPT being another version of tutorial hell with its limited "creativity" capacity in the probability for a string output, there is a limited space to grow and expand
deep concrete knowledge of artificial intelligence, however from my experience doing something from near scratch is an excellent way to understand the higher-level knowledge
that is introduced to beginners in something such as tensorflow.


This project aims to help guide beginners into developing a concrete foundational understanding of artificial intelligence, its underlying mathematics, the relative concepts, 
and important deep-level issues causing limitations in modern models such as global minima in gradient descent, escaping local minima points, avoiding dead gradients/dead neurons, ect. 

## Currently completed projects:
### Deep Neural Network
This network is the foundation for most architectures from here forward, a multi-layer neural network supporting different activation functions and an arbitrary amount of layers. This network should also (ideally) be the introduction to artificial intelligence beginners take as more complex network infrastructures can often revolve around solving problems with only using a deep neural network or improve features a deep neural network already introduces. The deep feed forward neural network from scratch notebook will go over multiple introductory concepts for artificial intelligence, its underlying mathematics, and how it can be implemented in code. The topics include:
* Data normalization
* Weight and bias initialization techniques
* Activation functions and their corresponding derivative functions
* layer initialization methods
* calculating the z state (weighted sum) and applying activation functions
* calculating loss metrics
* backpropagation, the relevant calculus, and gradients

  ![AIScratch_Deep Feed Forward Neural Network from scratch ipynb at main · Alt900_AIScratch — Mozilla Firefox 4_13_2024 10_38_09 PM](https://github.com/Alt900/AIScratch/assets/146238918/958eaced-3866-44e6-82e9-eb448e499bd1)
![AIScratch_Deep Feed Forward Neural Network from scratch ipynb at main · Alt900_AIScratch — Mozilla Firefox 4_13_2024 10_38_13 PM](https://github.com/Alt900/AIScratch/assets/146238918/4db636c6-b1e6-42a0-b7e1-43be31c753c7)


## Currently under construction
### Convolutional neural network
This network expands upon the DNN by processing images through convolution and max pooling operations to highlight certain features in an image to better guide the DNN. Currently the project has a completed forward pass with 2D Convolution using the fast fourier transform and the inverse fast fourier transform alongside minimum and maximum 2D pooling. As for backpropagation the 2D Convolution is handled through a slight modification of the DNN backpropagation methods to handle additional dimensions in the bias terms, as for pooling backpropagation the masks are generated to propagate the gradient through pooling and the images are loaded during backpropagation to tell the network where to propagate the error through
![f4_Conv1](https://github.com/Alt900/AIScratch/assets/146238918/20f91e34-49a5-43a2-9bed-09846dc8d773)
![f4_Conv2](https://github.com/Alt900/AIScratch/assets/146238918/a32e6a02-3039-4d16-868c-6c4d3ffa57e0)
![f4_MaxPooling2D_2](https://github.com/Alt900/AIScratch/assets/146238918/0d3b0fc3-c0f4-4df3-808d-ea282bc9a335)

![f1_Max_1_mask](https://github.com/Alt900/AIScratch/assets/146238918/3a50dedd-5ea4-47bb-a3d3-cb02f1e612f2)



## Future projects
### Recurrent neural network
### Long short term memory
### generative adversarial network
### transformer networks
