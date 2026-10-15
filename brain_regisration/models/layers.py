import keras
from keras import layers, ops, initializers
import tensorflow.keras.backend as K

# ReflectionPadding2D Layer
class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)
        

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return ops.pad(input_tensor, padding_tensor, mode="REFLECT")

# Residual Block
def residual_block(
    x,
    activation,
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    x = layers.add([input_tensor, x])
    return x

# Downsample Block
def downsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
    gamma_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    if activation:
        x = activation(x)
    return x

# Upsample Block
def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
    gamma_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def self_attention_block(x):
    # 从 x 的 shape 中取出通道数
    channels = x.shape[-1]
    
    # 1×1 卷积分别得到 Query (f)、Key (g)、Value (h)
    f = layers.Conv2D(channels // 8, 1, padding="same")(x)   # Query
    g = layers.Conv2D(channels // 8, 1, padding="same")(x)   # Key
    h = layers.Conv2D(channels,     1, padding="same")(x)   # Value

    # reshape → (batch, H*W, C') 以便后续点积
    f_flat = layers.Reshape((-1, channels // 8))(f)
    g_flat = layers.Reshape((-1, channels // 8))(g)
    h_flat = layers.Reshape((-1, channels))(h)

    # Attention map: softmax(Q · K^T)
    attn = layers.Softmax(axis=-1)(
        layers.Dot(axes=(2,2))([f_flat, g_flat])
    )  # (batch, H*W, H*W)

    # 加权求和：A · V → reshape 回 (batch, H, W, C)
    o = layers.Dot(axes=(2,1))([attn, h_flat])
    o = layers.Reshape((x.shape[1], x.shape[2], channels))(o)

    # 可学习缩放参数 γ，初始为 0
    gamma = K.zeros((1,))
    o = gamma * o + x
    # 残差连接
    return o

def dense_block(x, num_layers=4, growth_rate=32):
    features = [x]
    for i in range(num_layers):
        y = layers.Concatenate()(features)
        y = layers.GroupNormalization(groups=1)(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(growth_rate, (3,3), padding="same", use_bias=False)(y)
        features.append(y)
    return layers.Concatenate()(features)