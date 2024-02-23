import tensorflow.keras.layers as tfl
import tensorflow.keras as tfk

# CONSTANTS
DROPOUT = 0.2  # dropout coefficient
Kf = 0.25  # filters coefficient


# Model architecture
def downsample_block(input_tensor, filters):
    """Convolution"""
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(input_tensor)
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    skip = x
    x = tfl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
    return x, skip


def downsample_block_dropout(input_tensor, filters):
    """Last convolution in the block with a dropout"""
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(input_tensor)
    x = tfl.BatchNormalization()(x)
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.Dropout(DROPOUT)(x)  ## place?
    skip = x
    x = tfl.MaxPooling2D(strides=(2, 2), padding="valid")(x)  # pool_size=(2, 2),
    return x, skip


def upsample_block(input_tensor, skip_tensor, filters):
    """Deconvolution"""
    x = tfl.Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")(
        input_tensor
    )  #  // 2 at filters
    x = tfl.BatchNormalization()(x)
    x = tfl.concatenate([x, skip_tensor], axis=3)
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.Conv2D(
        filters,
        kernel_size=3,
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    return x


def unet_model(input_shape):
    """The layers"""
    inputs = tfk.Input(shape=input_shape)  # Load data

    # Downsample path
    down1, skip1 = downsample_block(inputs, Kf * 64)
    down2, skip2 = downsample_block(down1, Kf * 128)
    down3, skip3 = downsample_block(down2, Kf * 256)
    down4, skip4 = downsample_block_dropout(down3, Kf * 512)

    # Bridge
    bridge = tfl.Conv2D(Kf * 1024, 3, activation="elu", padding="same")(down4)
    bridge = tfl.BatchNormalization()(bridge)
    bridge = tfl.Conv2D(Kf * 1024, 3, activation="elu", padding="same")(bridge)
    bridge = tfl.Dropout(DROPOUT)(bridge)

    # Upsample path
    up4 = upsample_block(bridge, skip4, Kf * 512)
    up3 = upsample_block(up4, skip3, Kf * 256)
    up2 = upsample_block(up3, skip2, Kf * 128)
    up1 = upsample_block(up2, skip1, Kf * 64)

    # Output layer
    outputs = tfl.Conv2D(
        kernel_size=1,
        filters=1,
        activation="sigmoid",
        padding="same",
        kernel_initializer="he_normal",
    )(up1)

    model = tfk.Model(inputs=inputs, outputs=outputs)
    return model
