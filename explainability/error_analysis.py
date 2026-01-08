import numpy as np
from gradcam import make_gradcam_heatmap, display_gradcam

def analyze_errors(model, X_test, y_test, class_names):
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    errors = np.where(pred_labels != true_labels)[0]

    for idx in errors[:5]:
        heatmap = make_gradcam_heatmap(
            X_test[idx:idx+1],
            model,
            "conv5_block3_out"
        )
        display_gradcam(X_test[idx], heatmap)
