import numpy as np
from tensorflow.image import resize_with_pad
from .build_model import build_model


def resize_images(images, size=224):
    return np.array([
        resize_with_pad(img, size, size) for img in images
    ]).astype("float32")


def train_model(X_train, y_train, X_test, y_test, epochs=50):
    model = build_model()

    X_train = resize_images(X_train * 255.0)
    X_test = resize_images(X_test * 255.0)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs
    )

    model.save("checkpoints/trained_model.h5")
    return model
