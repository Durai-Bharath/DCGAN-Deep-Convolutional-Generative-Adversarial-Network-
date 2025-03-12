from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model

class  DCGAN:
    @staticmethod
    def build_generator(dim,depth,channels = 1,inputDim = 100, outputDim = 512):
        #where dim is the spatial dimension of the input volume (i.e., the width and height of the input image)
        #depth is the number of filters in the first layer of the generator(we will double the number of filters in each subsequent layer)
        
        model = Sequential()
        inputShape = (dim,dim,depth) #input shape of the generator
        chanDim = -1 #ChanDim is the channel dimension meaning that the data is in the format of (batch, height, width, channels)
                     #1 for grayscale images and 3 for RGB images
        
        
        #first set of FC => RELU => BN layers
        model.add(Dense(input_dim= inputDim,units=outputDim)) #fully connected layer where units is the number of nodes in the layer
        model.add(Activation("relu") ) #activation function relu
        model.add(BatchNormalization()) #batch normalization meaning that we normalize the activations of the network at each layer divided into mini-batches
        
        #second set of FC => RELU => BN layers 
        #this time the number of FC nodes to be reshaped into a volume becasue we are going to apply a convolutional transpose layer where input_dim < output_dim
        
        model.add(Dense(dim*dim*depth)) #fully connected layer has output of dim*dim*depth
        model.add(Activation("relu")) #activation function relu
        model.add(BatchNormalization()) #batch normalization    
        
        #reshape the volume to have the desired spatial dimensions
        model.add(Reshape(inputShape)) #reshape the volume to have the desired spatial dimensions
        model.add(Conv2DTranspose(32,(5,5),strides=(2,2),padding="same")) #convolutional transpose layer where 32 is the number of filters and (5,5) is the size of the filter and strides is the step size of the filter
        model.add(Activation("relu")) #activation function relu
        model.add(BatchNormalization(axis=chanDim)) #batch normalization where chanDim is -1 means it takes the last dimension of the input
        
        #apply another convolutional transpose layer
        model.add(Conv2DTranspose(channels,(5,5),strides=(2,2),padding="same")) #convolutional transpose layer where channels is the number of filters and (5,5) is the size of the filter and strides is the step size of the filter
        model.add(Activation("tanh")) #activation function tanh
        
        
        return model
    
    @staticmethod
    def build_discriminator(width,height,depth,alpha=0.2):
        #where width and height are the spatial dimensions of the input volume
        #depth is the number of filters in the first layer of the discriminator
        #alpha is the slope of the leaky ReLU activation function in simple terms it is the learning rate of the activation function
        
        model = Sequential()
        inputShape = (height,width,depth)
        
        #first set of CONV => RELU layers
        model.add(Conv2D(32,(5,5),padding="same",input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))
        
        
        #second set of CONV => RELU layers  
        model.add(Conv2D(64,(5,5),strides = (2,2),padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        
        #third set Flatten => FC layer (fully connected layer)
        model.add(Flatten()) #flatten the volume into a vector as the input to the fully connected layer in column major order
        model.add(Dense(512)) #512 is the number of nodes in the layer
        model.add(LeakyReLU(alpha=alpha))
        
        #output layer
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        return model
        
        
