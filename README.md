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


## Currently under construction
### Convolutional neural network
This network expands upon the DNN by processing images through convolution and max pooling operations to highlight certain features in an image to better guide the DNN. Currently the project has a completed forward pass with 2D Convolution using the fast fourier transform and the inverse fast fourier transform alongside minimum and maximum 2D pooling. As for backpropagation the 2D Convolution is handled through a slight modification of the DNN backpropagation methods to handle additional dimensions in the bias terms, as for pooling backpropagation the masks are generated to propagate the gradient through pooling and the images are loaded during backpropagation to tell the network where to propagate the error through



## Future projects
### Recurrent neural network
### Long short term memory
### generative adversarial network
### transformer networks
