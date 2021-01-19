import numpy as np
import keras.models 
from keras.models import Model
import matplotlib.pyplot as plt
import keras.backend as K
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.fc2.register_forward_hook(get_activation('conv1'))
x = torch.randn(1, 25)
output = model(x)
print(activation['fc2'])
class GradCAM:
    def __init__(self, model):
        maskh0=self.getGradCamMask(model.classifier.output[0,0],model.conv3.output) #activation layer 
        maskh1=self.getGradCamMask(model.classifier.output[0,1],model.conv3.output)
        getMasks=K.function([model.conv1.input,model.conv2.input],[maskh0,maskh1])
        self.getMasks = getMasks
        
    '''def getGradCamMask(self,output,activation):

        This function calculates the importance weight reported in GradCam
        Input:
            ouput: The class output 
            activation: activation that we will take gradient with respect to

        grad=K.gradients(output,activation)[0]
        alpha=K.squeeze(K.mean(grad,axis=1),0)
        mask=K.squeeze(K.relu(K.sum(activation*alpha,axis=2)),0)
        return mask
    '''
    def getGradCamMask(self,output,activation):
        '''
        This function calculates the importance weight reported in GradCam
        Input:
            ouput: The class output 
            activation: activation that we will take gradient with respect to
        '''
        grad = torch.autograd.grad(output,activation)[0]
        alpha = torch.squeeze(torch.math.mean(grad,axis=1),0)
        mask = torch.squeeze(torch.nn.ReLU(torch.math.sum(activation*alpha,axis = 2)),0)
class GradCAMAvg:
    
    def __init__(self, model):
        maskInput0=self.getGradCamMask(model.layers[-2].output[0,0],model.layers[1].input)
        maskInput1=self.getGradCamMask(model.layers[-2].output[0,1],model.layers[1].input)
        tempMax0=K.max(K.stack([maskInput0,maskInput1]))

        mask0h1=self.getGradCamMask(model.layers[-2].output[0,0],model.layers[-9].output)
        mask1h1=self.getGradCamMask(model.layers[-2].output[0,1],model.layers[-9].output)
        tempMax1=K.max(K.stack([mask0h1,mask1h1]))

        mask0h2=self.getGradCamMask(model.layers[-2].output[0,0],model.layers[-7].output)
        mask1h2=self.getGradCamMask(model.layers[-2].output[0,1],model.layers[-7].output)
        tempMax2=K.max(K.stack([mask0h2,mask1h2]))

        mask0h3=self.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        mask1h3=self.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        tempMax3=K.max(K.stack([mask0h3,mask1h3]))

        getMasks=K.function([model.layers[0].input,model.layers[1].input],[maskInput0/tempMax0+mask0h1/tempMax1+mask0h2/tempMax2
                                                                           +mask0h3/tempMax3,maskInput1/tempMax0+mask1h1/tempMax1+
                                                                           mask1h2/tempMax2+mask1h3/tempMax3])
        self.getMasks = getMasks
    
    def getGradCamMask(self,output,activation):
        '''
        This function calculates the importance weight reported in GradCam
        Input:
            ouput: The class output 
            activation: activation that we will take gradient with respect to
        '''
        grad=K.gradients(output,activation)[0]
        alpha=K.squeeze(K.mean(grad,axis=1),0)
        mask=K.squeeze(K.relu(K.sum(activation*alpha,axis=2)),0)
        return mask