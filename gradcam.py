import cv2
import numpy as np
import tensorflow as tf

def get_last_conv_layer_name(model):
    """Finds the name of the final convolutional layer in the base model."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):  # Check nested model (e.g., EfficientNetB0)
            for sub_layer in reversed(layer.layers):
                if 'conv' in sub_layer.name or 'activation' in sub_layer.name:
                    return sub_layer.name
        elif 'conv' in layer.name:
            return layer.name
    return "top_conv"


def compute_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Computes a Grad-CAM heatmap using GradientTape directly
    targeting the last 4D activation tensor.
    """
    # 1. Target the nested base feature extractor
    base_model = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() or hasattr(layer, 'layers'):
            base_model = layer
            break

    # 2. Determine target class index if not provided
    preds = model.predict(img_array, verbose=0)
    if pred_index is None:
        pred_index = int(np.argmax(preds[0]))

    # 3. Build sub-model to tap the last conv layer output
    if base_model is not None:
        last_conv_name = get_last_conv_layer_name(base_model)
        last_conv_layer = base_model.get_layer(last_conv_name)
        
        # Sub-model mapping base_model inputs -> last conv feature maps
        feature_model = tf.keras.models.Model(
            inputs=[base_model.inputs],
            outputs=[last_conv_layer.output]
        )

        with tf.GradientTape() as tape:
            # Pass through base model conv layers
            conv_features = feature_model(img_array)
            tape.watch(conv_features)

            # Pass remaining layers manually
            x = conv_features
            # Apply layers that sit after base_model in main model
            found_base = False
            for layer in model.layers:
                if found_base:
                    x = layer(x)
                elif layer == base_model:
                    found_base = True

            loss = x[:, pred_index]

        # Gradients of target class with respect to conv_features
        grads = tape.gradient(loss, conv_features)
    else:
        # Flat model structure
        last_conv_name = get_last_conv_layer_name(model)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_features, preds_out = grad_model(img_array)
            loss = preds_out[:, pred_index]

        grads = tape.gradient(loss, conv_features)

    # 4. Global Average Pooling of gradients (neuron importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weighted combination of feature maps
    conv_features = conv_features[0]
    heatmap = conv_features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (we only care about positive influences on this defect class)
    heatmap = tf.maximum(heatmap, 0.0)

    # 7. Normalize between 0 and 1 with numerical stability
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    else:
        # Fallback if zero gradients
        heatmap = tf.ones_like(heatmap)

    return heatmap.numpy()


def generate_and_save_gradcam(original_img_path, processed_tensor, model, output_path, pred_index=None):
    try:
        # Generate the raw heatmap array
        heatmap = compute_gradcam_heatmap(processed_tensor, model, pred_index=pred_index)
        
        # Load original image
        img = cv2.imread(original_img_path)
        if img is None:
            return None

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Contrast stretch: increase visibility of peak defect zones
        heatmap_resized = np.power(heatmap_resized, 0.8)  # slight gamma expansion
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap (Red = strongest defect activation, Blue = background)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay on the original image: 65% original + 35% thermal heatmap
        overlay = cv2.addWeighted(img, 0.65, heatmap_colored, 0.35, 0)

        # Draw a subtle border around highest activation spots
        cv2.imwrite(output_path, overlay)
        return output_path
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None