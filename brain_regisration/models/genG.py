import keras
from keras import layers
from keras import ops
import tensorflow as tf
from .layers import ReflectionPadding2D, residual_block, downsample, upsample, self_attention_block, dense_block


#跳跃连接
def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=3,
    num_residual_blocks=15,
    num_upsample_blocks=3,
    gamma_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
    name=None,
):
    img_input = layers.Input(shape=(256, 256, 1), name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False)(
        x
    )
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling + Skip Connections
    skip_connections = []
    for _ in range(num_downsampling_blocks):
        filters *= 2
        skip_connections.append(x)  # Save the feature map for later use
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # 2) 残差阶段前半
    for _ in range(num_residual_blocks // 2):
        x = residual_block(x, activation=layers.Activation("relu"))

    # —— 在这里插入 Self-Attention —— 
    x = dense_block(x, num_layers=4, growth_rate=32)
    x = self_attention_block(x)

    # 3) 残差阶段后半
    for _ in range(num_residual_blocks // 2, num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling with Skip Connections
    for skip in reversed(skip_connections):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))
        x = layers.Concatenate()([x, skip])  # Add skip connection

    # Final block
    x = layers.GroupNormalization(groups=1)(x)
    x = layers.Activation("relu")(x)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(1, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model