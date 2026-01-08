from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model


def build_model(num_classes=8):
    base_model = ResNet50(include_top=True, weights="imagenet")

    x = base_model.layers[-2].output
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(base_model.input, output)

    # Freeze backbone
    for layer in model.layers:
        layer.trainable = False

    # Fine-tune last layers
    for layer in model.layers[-3:]:
        layer.trainable = True

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
