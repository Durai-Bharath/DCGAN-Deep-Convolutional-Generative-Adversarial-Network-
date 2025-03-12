from pyimagesearch.dcgan import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,
                help="path to output directory")
ap.add_argument("-e","--epochs",type=int,default=50,
                help="# of epochs to train for")
ap.add_argument("-b","--batch-size",type=int,default=128,
                help="size of mini-batches for training")
args = vars(ap.parse_args())


# initialize the learning rate and batch size
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4

# load the MNIST dataset
print("[INFO] loading MNIST dataset...")
((trainX,_),(testX,_)) = mnist.load_data()

trainImages = np.concatenate([trainX,testX])

# add in an extra dimension for the channel and scale the images to the range [-1,1]
# the generator will use the tanh activation function while the discriminator will use the sigmoid activation function
trainImages = np.expand_dims(trainImages , axis = -1) # axis = -1 add an extra dimension at the end of the array
trainImages = (trainImages.astype("float" ) - 127.5) / 127.5 # scale the images to the range [-1,1]


print("[INFO] building the generator...")
gen = DCGAN.build_generator(7,64,channels = 1)

print("[INFO] building the discriminator...")
disc = DCGAN.build_discriminator(28,28,1)

#Add optimizers to the discriminator
disOpt = Adam(lr=INIT_LR,beta_1=0.5,decay = INIT_LR / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy",optimizer=disOpt)


# build the adversarial model by first setting the discriminator to *not* be trainable, then combine the generator and discriminator together
print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))#input to the generator where 100 is the dimension of the input vector
ganOutput = disc(gen(ganInput)) #connect the generator to the discriminator
gan = Model(ganInput,ganOutput) #combine the generator and discriminator together

# compile the GAN
ganOpt = Adam(lr=INIT_LR,beta_1=0.5,decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy",optimizer=ganOpt)


# randomly generate some benchmark noise so we can consistently visualize how the generative modeling is learning
#visualize the progress of the GAN
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1,1,size=(256,100)) #generate 256 random numbers between -1 and 1 where 100 is the dimension of the input vector

# loop over the epochs
for epoch in range(0,NUM_EPOCHS):
    # show epoch information and compute the number of batches per epoch
    print("[INFO] starting epoch {} of {}...".format(epoch + 1,NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE) #number of batches per epoch
    
    # loop over the batches
    for i in range(0,batchesPerEpoch):
        # initialize an (empty) output path
        p = None
        
        #Select the next batch of images , then randomly generate noise for the generator to predict on(generate images)
        imageBatch = trainImages[i*BATCH_SIZE:(i+1)*BATCH_SIZE] #select the next batch of images where : is the range operator it selects the range of pixels from i*BATCH_SIZE to (i+1)*BATCH_SIZE
        noise = np.random.uniform(-1,1,size=(BATCH_SIZE,100)) #generate random noise for the generator to predict on
        #Given the noise vector to the generator to generate synthetic images
        
        # generate images using the noise + generator model
        genImages = gen.predict(noise,verbose=0) #where verbose = 0 means no output is displayed
        
        #concatenate the "actual" images and the "generated" images ,
        #construct class labels for the discriminator and shuffle the data
        X = np.concatenate((imageBatch,genImages)) #concatenate the actual images and the generated images
        y = ([1] * BATCH_SIZE) + ([0]*BATCH_SIZE) #construct class labels for the discriminator where 1 is the actual image and 0 is the generated image
        y = np.reshape(y,(-1,1)) #in the form of a column vector
        (X,y) = shuffle(X,y) #shuffle the data to prevent the discriminator from overfitting
        
        #train the discriminator on the data 
        discLoss = disc.train_on_batch(X,y) #train the discriminator on the data where train_on_batch is a inbuilt function in keras and it returns the loss of the model and train the model on a single batch of data for the given number of epochs
        
        #train generator via the adversarial model where the discriminator is not trainable
        #generating random noise and training the generator with the discriminator weights frozen
        noise = np.random.uniform(-1,1,(BATCH_SIZE,100)) #generate random noise for the generator to predict on
        fakeLabels = [1] * BATCH_SIZE #construct class labels for the generator where 1 is the actual image
        fakeLabels = np.reshape(fakeLabels,(-1,1)) #in the form of a column vector
        ganLoss = gan.train_on_batch(noise,fakeLabels) #train the generator on the data where train_on_batch is a inbuilt function in keras and it returns the loss of the model and train the model on a single batch of data for the given number of epochs
        
        
        #check to see if this is the end of an epoch , and if so , initialize the output path
        if i == batchesPerEpoch - 1:
            p = [args["output"],"epoch_{}_output.png".format(str(epoch + 1).zfill(4))] #initialize the output path where zfill(4) means that the string is padded with zeros to ensure that it is 4 characters long
            
        #otherwise , check to see if we should visualize the current batch for the epoch
        else:
            #create more visualizations for the GAN
            if epoch < 10 or (i % 25 == 0):
                p = [args["output"],"epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4),str(i).zfill(5))]
            
            elif epoch > 10 and i % 100 == 0:
                p = [args["output"],"epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4),str(i).zfill(5))]                
        
        
        #check to see if we should visualize the output of the generator model on our benchmark data
        if p is not None:
            # show loss information
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}".format(epoch + 1,i,discLoss,ganLoss))
            
            # make predictions on the benchmark noise, scale it back to the range [0,255], and generate the montage
            images = gen.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images , 3 , axis = -1) #repeat the images 3 times along the last axis 
            vis = build_montages(images,(28,28),(16,16))[0] # build the montages of the images where 28,28 is the size of the image and 16,16 is the number of images in the montage
            
            # write the visualization to disk
            p = os.path.sep.join(p)
            cv2.imwrite(p,vis)