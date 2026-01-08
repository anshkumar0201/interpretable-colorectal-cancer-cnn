import numpy as np
from .gradcam import make_gradcam_heatmap, display_gradcam


def analyze_errors(model, X_test, y_test, class_names):
    preds = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    errors = np.where(pred_labels != true_labels)[0]

    if len(errors) == 0:
        print("No misclassifications found.")
        return

    for idx in errors[:5]:
        print(
            f"Misclassified sample {idx}: "
            f"true={class_names[true_labels[idx]]}, "
            f"predicted={class_names[pred_labels[idx]]}"
        )

        heatmap = make_gradcam_heatmap(
            X_test[idx:idx + 1],
            model,
            "conv5_block3_out"
        )

        display_gradcam(X_test[idx], heatmap)
