# source: DeepLearning_code_UNet_ResNet_v3.ipnb

import tensorflow as tf
from tensorflow.keras import layers

img_size = (64,64)

def dBlock(x, n_filters):
    # BatchNorm
    y = layers.BatchNormalization()(x)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Conv(3,2)
    y = layers.Conv2D(n_filters, 3, strides=2, padding="same")(y)
    # BatchNorm
    y = layers.BatchNormalization()(y)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Conv(3,1)
    y = layers.Conv2D(n_filters, 3, padding="same")(y)

    # Skip connection
    skip = layers.Conv2D(n_filters, 3, strides=2, padding="same")(x)
    
    # Element-wise addition
    x = layers.Add()([y, skip])
    return x


def uBlock(x, n_filters):
    # BatchNorm
    y = layers.BatchNormalization()(x)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Transposed convolution
    y = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(y)
    # BatchNorm
    y = layers.BatchNormalization()(y)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Conv(3,1)
    y = layers.Conv2D(n_filters, 3, padding="same")(y)

    # Skip connection
    skip = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
    
    # Element-wise addition
    x = layers.Add()([y, skip])
    return x

def sBlock(x, n_filters):
    # Conv(3,1)
    x = layers.Conv2D(n_filters, 3, padding="same")(x)
    # BatchNorm
    x = layers.BatchNormalization()(x)
    # ReLU activation
    x = layers.Activation('relu')(x)
    # Conv(3,1)
    x = layers.Conv2D(n_filters, 3, padding="same")(x)
    # BatchNorm
    x = layers.BatchNormalization()(x)
    # ReLU activation
    x = layers.Activation('relu')(x)
    return x

def rSubBlock(x, n_filters):
    # BatchNorm
    y = layers.BatchNormalization()(x)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Conv(3,1)
    y = layers.Conv2D(n_filters, 3, padding="same")(y)
    # BatchNorm
    y = layers.BatchNormalization()(y)
    # ReLU activation
    y = layers.Activation('relu')(y)
    # Conv(3,1)
    y = layers.Conv2D(n_filters, 3, padding="same")(y)
    
    # Skip connection (without any convolution layer) and Element-wise addition
    x = layers.Add()([y, x])
    #x = layers.concatenate([y, x])
    return x

def unet_model():    
    
    inputs = layers.Input(shape=img_size + (3,))

    # down-sampling
    s1 = sBlock(inputs, 32)
    d1 = dBlock(inputs, 32)
    
    s2 = sBlock(d1, 32)
    d2 = dBlock(d1, 32)
    
    s3 = sBlock(d2, 32)
    d3 = dBlock(d2, 32)
    
    s4 = sBlock(d3, 32)
    d4 = dBlock(d3, 32)
    
    s5 = sBlock(d4, 32)
    d5 = dBlock(d4, 32)
    
    s6 = sBlock(d5, 32)
    d6 = dBlock(d5, 32)
    #---------------------------------------------
    u6= uBlock(d6,32)
    #---------------------------------------------
    # up-sampling
    u6 = layers.concatenate([u6, s6])
    #u6 = layers.Add()([u6, s6])
    
    u5= uBlock(u6,32)
    u5 = layers.concatenate([u5, s5])
        
    u4= uBlock(u5,32)
    u4 = layers.concatenate([u4, s4])
        
    u3= uBlock(u4,32)
    u3 = layers.concatenate([u3, s3])
    
    u2 = uBlock(u3,32)
    u2 = layers.concatenate([u2, s2])
    
    u1 = uBlock(u2,32)
    u1 = layers.concatenate([u1, s1])
    
    r1 = rSubBlock(u1, 64)
    outputs = layers.Conv2D(3, 3, padding="same", activation="relu")(r1)
    
    # Define the model
    
    model = tf.keras.Model(inputs, outputs)
    return model

model = unet_model()
model.summary()

#plot_model(model, to_file='model_diagram.png', show_shapes=True, rankdir='TB')
# Display the saved image in the notebook
#Image('model_diagram.png')