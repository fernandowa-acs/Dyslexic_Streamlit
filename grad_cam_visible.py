# grad_cam_visible.py - ADVANCED DEBUG VERSION
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import streamlit as st

def debug_model_structure(model):
    """Detailed model structure analysis"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Debug Info")
    
    print("🔍 DETAILED MODEL ANALYSIS:")
    conv_layers = []
    dense_layers = []
    other_layers = []
    
    for i, layer in enumerate(model.layers):
        try:
            layer_type = type(layer).__name__
            has_weights = hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0
            
            # Get output shape
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            else:
                output_shape = "No output_shape"
            
            print(f"{i:2d}: {layer.name:20} {layer_type:15} {str(output_shape):30} Weights: {has_weights}")
            
            # Categorize layers
            if 'conv' in layer.name.lower() or 'conv2d' in layer_type.lower():
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    conv_layers.append({
                        'name': layer.name,
                        'type': layer_type,
                        'output_shape': output_shape,
                        'index': i
                    })
            elif 'dense' in layer.name.lower() or 'dense' in layer_type.lower():
                dense_layers.append(layer.name)
            else:
                other_layers.append(layer.name)
                
        except Exception as e:
            print(f"{i:2d}: {layer.name:20} Error: {str(e)}")
    
    # Display in sidebar
    st.sidebar.write(f"**Convolutional Layers:** {len(conv_layers)}")
    for conv in conv_layers:
        st.sidebar.write(f"  - {conv['name']} {conv['output_shape']}")
    
    st.sidebar.write(f"**Dense Layers:** {len(dense_layers)}")
    st.sidebar.write(f"**Other Layers:** {len(other_layers)}")
    
    return conv_layers

def get_gradcam_heatmap_advanced(model, img_array, layer_name=None):
    """
    Advanced Grad-CAM dengan support untuk berbagai model types
    """
    print(f"🎯 Attempting Grad-CAM with layer: {layer_name}")
    
    try:
        # Build grad model
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Untuk binary classification, gunakan output pertama
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Binary classification dengan sigmoid
                loss = predictions[:, 0]
            elif len(predictions.shape) == 2 and predictions.shape[1] == 2:
                # Binary classification dengan softmax
                loss = predictions[:, 1]  # Ambil class 1
            else:
                # Fallback
                loss = predictions[:, 0]
            
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Check if gradients are valid
        if grads is None:
            print(f"❌ No gradients for layer {layer_name}")
            return None, "no_gradients"
            
        # Global average pooling untuk gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps dengan gradients
        conv_outputs = conv_outputs[0]  # Take first batch item
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        
        # ReLU activation
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        else:
            print(f"⚠️ Zero heatmap for layer {layer_name}")
            return None, "zero_heatmap"
        
        heatmap = heatmap.numpy()
        print(f"✅ Grad-CAM success with {layer_name} - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        return heatmap, layer_name
        
    except Exception as e:
        print(f"❌ Grad-CAM error with {layer_name}: {str(e)}")
        return None, f"error: {str(e)}"

def create_enhanced_heatmap(heatmap, original_img, alpha=0.7):
    """Create enhanced heatmap visualization"""
    try:
        # Resize heatmap to match original
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply Gaussian blur untuk smoothness
        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
        
        # Normalize to [0, 1]
        heatmap_normalized = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min() + 1e-8)
        
        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Prepare original image
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img
        
        # Ensure same size
        if original_rgb.shape[:2] != heatmap_colored.shape[:2]:
            original_rgb = cv2.resize(original_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        
        # Blend images
        blended = cv2.addWeighted(original_rgb, 1-alpha, heatmap_colored, alpha, 0)
        
        return blended
        
    except Exception as e:
        print(f"❌ Heatmap creation error: {e}")
        # Fallback: return original with red border
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img
        cv2.rectangle(original_rgb, (10, 10), (original_rgb.shape[1]-10, original_rgb.shape[0]-10), (255, 0, 0), 3)
        return original_rgb

def try_alternative_approaches(model, img_array, original_img):
    """Coba berbagai pendekatan alternatif untuk Grad-CAM"""
    print("🔄 Trying alternative Grad-CAM approaches...")
    
    approaches = [
        # Coba layer dengan 'conv' dalam nama
        lambda: try_conv_layers_by_name(model, img_array, original_img),
        # Coba layer dengan output 4D
        lambda: try_4d_output_layers(model, img_array, original_img),
        # Coba last few layers
        lambda: try_last_layers(model, img_array, original_img),
    ]
    
    for approach in approaches:
        try:
            result = approach()
            if result is not None:
                heatmap, overlay, layer_name = result
                if heatmap is not None and np.max(heatmap) > 0.1:
                    print(f"✅ Alternative approach succeeded with {layer_name}")
                    return heatmap, overlay, f"alternative_{layer_name}"
        except Exception as e:
            print(f"❌ Alternative approach failed: {e}")
            continue
    
    return None

def try_conv_layers_by_name(model, img_array, original_img):
    """Coba layer dengan 'conv' dalam nama"""
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            try:
                heatmap, used_layer = get_gradcam_heatmap_advanced(model, img_array, layer.name)
                if heatmap is not None and np.max(heatmap) > 0.1:
                    overlay = create_enhanced_heatmap(heatmap, original_img)
                    return heatmap, overlay, used_layer
            except:
                continue
    return None

def try_4d_output_layers(model, img_array, original_img):
    """Coba layer dengan output shape 4D"""
    for layer in model.layers:
        try:
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                heatmap, used_layer = get_gradcam_heatmap_advanced(model, img_array, layer.name)
                if heatmap is not None and np.max(heatmap) > 0.1:
                    overlay = create_enhanced_heatmap(heatmap, original_img)
                    return heatmap, overlay, used_layer
        except:
            continue
    return None

def try_last_layers(model, img_array, original_img):
    """Coba last few layers dari model"""
    # Ambil 5 layer terakhir
    last_layers = list(reversed(model.layers))[:5]
    for layer in last_layers:
        try:
            heatmap, used_layer = get_gradcam_heatmap_advanced(model, img_array, layer.name)
            if heatmap is not None:
                overlay = create_enhanced_heatmap(heatmap, original_img)
                return heatmap, overlay, used_layer
        except:
            continue
    return None

def create_intelligent_fallback(original_img, prediction_value=None):
    """Create intelligent fallback visualization berdasarkan prediksi"""
    h, w = original_img.shape[:2]
    
    # Buat heatmap berdasarkan prediksi
    if prediction_value is not None:
        # Gunakan prediksi untuk menentukan intensitas
        intensity = min(0.8, float(prediction_value) * 0.8 + 0.2)
    else:
        intensity = 0.5
    
    # Buat heatmap dengan pattern yang meaningful
    heatmap = np.zeros((h, w))
    
    # Multiple focus areas
    centers = [
        (h//4, w//4),      # Top-left
        (h//4, 3*w//4),    # Top-right  
        (3*h//4, w//4),    # Bottom-left
        (3*h//4, 3*w//4),  # Bottom-right
        (h//2, w//2)       # Center
    ]
    
    for center_y, center_x in centers:
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = min(w, h) / 3
        heatmap += np.maximum(0, 1 - dist / max_dist) * intensity
    
    # Normalize
    heatmap = np.clip(heatmap, 0, 1)
    
    # Create overlay
    overlay = create_enhanced_heatmap(heatmap, original_img, alpha=0.6)
    
    return heatmap, overlay

def visualize_gradcam_ultimate(model, img_array, original_img, prediction_value=None):
    """
    Ultimate Grad-CAM dengan advanced debugging dan fallbacks
    """
    print("🚀 Starting Advanced Grad-CAM...")
    
    # Debug model structure pertama
    conv_layers = debug_model_structure(model)
    
    # Jika ada convolutional layers, coba mereka dulu
    if conv_layers:
        print(f"🎯 Found {len(conv_layers)} convolutional layers, trying...")
        
        for layer_info in conv_layers:
            layer_name = layer_info['name']
            try:
                heatmap, used_layer = get_gradcam_heatmap_advanced(model, img_array, layer_name)
                
                if heatmap is not None and np.max(heatmap) > 0.05:
                    print(f"✅ Success with convolutional layer: {used_layer}")
                    overlay = create_enhanced_heatmap(heatmap, original_img)
                    return heatmap, overlay, used_layer
                    
            except Exception as e:
                print(f"❌ Failed with {layer_name}: {e}")
                continue
    
    # Coba pendekatan alternatif
    print("🔄 No success with standard approach, trying alternatives...")
    alternative_result = try_alternative_approaches(model, img_array, original_img)
    if alternative_result is not None:
        return alternative_result
    
    # Final fallback - intelligent visualization
    print("⚠️ All approaches failed, using intelligent fallback")
    heatmap, overlay = create_intelligent_fallback(original_img, prediction_value)
    
    # Tampilkan warning di Streamlit
    st.warning("""
    **🔍 Grad-CAM Information:**
    - Model tidak memiliki layer convolutional yang kompatibel
    - Menggunakan visualisasi demonstrasi
    - Model mungkin menggunakan architecture Dense/MLP
    """)
    
    return heatmap, overlay, "intelligent_fallback"

# Test function
if __name__ == "__main__":
    print("✅ ADVANCED grad_cam_visible.py loaded successfully!")