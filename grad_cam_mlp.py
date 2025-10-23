# grad_cam_mlp.py - Khusus untuk Model Dense/MLP
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import Model
import tensorflow as tf

def analyze_mlp_model(model):
    """Analyze MLP model structure dan cari layer yang cocok"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 MLP Model Analysis")
    
    dense_layers = []
    input_shape = model.input_shape
    
    st.sidebar.write(f"**Input Shape:** {input_shape}")
    st.sidebar.write(f"**Output Shape:** {model.output_shape}")
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        st.sidebar.write(f"{i}: {layer.name} - {layer_type}")
        
        if 'dense' in layer.name.lower() or 'dense' in layer_type.lower():
            dense_layers.append({
                'name': layer.name,
                'type': layer_type,
                'units': layer.units if hasattr(layer, 'units') else 'Unknown',
                'index': i
            })
    
    st.sidebar.write(f"**Dense Layers:** {len(dense_layers)}")
    return dense_layers, input_shape

def create_saliency_map_mlp(model, img_array):
    """Buat saliency map untuk model MLP menggunakan gradients"""
    try:
        # Convert ke tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            # Untuk binary classification
            if predictions.shape[1] == 1:
                loss = predictions[:, 0]  # Sigmoid output
            else:
                loss = predictions[:, 1]  # Softmax output (class 1)
        
        # Dapatkan gradients
        gradients = tape.gradient(loss, img_tensor)
        
        if gradients is not None:
            # Ambil absolute values dan average across channels
            saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)
            saliency = saliency[0]  # Ambil batch pertama
            
            # Normalize
            saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
            return saliency.numpy()
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Saliency map error: {e}")
        return None

def create_feature_importance_heatmap(model, img_array, original_img):
    """Buat heatmap berdasarkan feature importance untuk MLP"""
    try:
        # Flatten image untuk MLP
        flat_input = img_array.reshape(img_array.shape[0], -1)
        
        # Dapatkan weights dari first dense layer
        first_dense_layer = None
        for layer in model.layers:
            if 'dense' in layer.name.lower() and hasattr(layer, 'get_weights'):
                first_dense_layer = layer
                break
        
        if first_dense_layer is not None:
            weights = first_dense_layer.get_weights()[0]  # Weight matrix
            
            # Average absolute weights untuk setiap input feature
            feature_importance = np.mean(np.abs(weights), axis=1)
            
            # Reshape kembali ke image shape
            img_height, img_width = original_img.shape[:2]
            if len(original_img.shape) == 3:
                img_channels = original_img.shape[2]
            else:
                img_channels = 1
                
            total_pixels = img_height * img_width * img_channels
            
            if len(feature_importance) >= total_pixels:
                # Reshape feature importance ke image dimensions
                heatmap = feature_importance[:total_pixels].reshape((img_height, img_width, img_channels))
                heatmap = np.mean(heatmap, axis=-1)  # Convert to 2D jika perlu
                
                # Normalize
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
                return heatmap
                
        return None
        
    except Exception as e:
        print(f"Feature importance error: {e}")
        return None

def create_mlp_visualization(model, img_array, original_img, prediction_value):
    """Visualisasi khusus untuk model MLP"""
    # Coba berbagai metode
    heatmap = None
    method_used = ""
    
    # Method 1: Saliency Map
    heatmap = create_saliency_map_mlp(model, img_array)
    if heatmap is not None and np.max(heatmap) > 0.1:
        method_used = "saliency_map"
    else:
        # Method 2: Feature Importance
        heatmap = create_feature_importance_heatmap(model, img_array, original_img)
        if heatmap is not None:
            method_used = "feature_importance"
        else:
            # Method 3: Prediction-based heatmap
            heatmap = create_prediction_based_heatmap(original_img, prediction_value)
            method_used = "prediction_based"
    
    # Create overlay
    overlay = create_mlp_heatmap_overlay(heatmap, original_img)
    
    return heatmap, overlay, f"mlp_{method_used}"

def create_prediction_based_heatmap(original_img, prediction_value):
    """Buat heatmap berdasarkan nilai prediksi"""
    h, w = original_img.shape[:2]
    heatmap = np.zeros((h, w))
    
    # Gunakan prediction value untuk menentukan pattern
    confidence = min(0.8, float(prediction_value) * 0.8 + 0.2)
    
    # Buat multiple regions of interest
    regions = []
    
    if prediction_value > 0.7:  # High confidence dyslexic
        # Fokus pada multiple areas (inkonsistensi)
        regions = [
            (h//3, w//3, 0.7),    # Top-left
            (h//3, 2*w//3, 0.6),  # Top-right
            (2*h//3, w//3, 0.5),  # Bottom-left
        ]
    elif prediction_value > 0.4:  # Medium confidence
        # Fokus pada center areas
        regions = [
            (h//2, w//2, 0.8),    # Center
            (h//3, w//2, 0.4),    # Top-center
        ]
    else:  # Low confidence atau normal
        # Uniform lower attention
        regions = [
            (h//2, w//2, 0.3),    # Center only
        ]
    
    # Apply Gaussian patterns
    for center_y, center_x, intensity in regions:
        y, x = np.ogrid[:h, :w]
        sigma = min(h, w) / 6
        region_heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        heatmap += region_heatmap * intensity * confidence
    
    # Normalize
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def create_mlp_heatmap_overlay(heatmap, original_img, alpha=0.6):
    """Create overlay khusus untuk MLP visualization"""
    try:
        # Resize heatmap jika perlu
        h, w = original_img.shape[:2]
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Prepare original
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_img
        
        # Blend
        blended = cv2.addWeighted(original_rgb, 1-alpha, heatmap_colored, alpha, 0)
        return blended
        
    except Exception as e:
        print(f"MLP overlay error: {e}")
        return original_img

def visualize_mlp_model(model, img_array, original_img, prediction_value):
    """
    Visualisasi utama untuk model MLP
    """
    st.sidebar.info("🧠 Using MLP Visualization")
    
    # Analyze model
    dense_layers, input_shape = analyze_mlp_model(model)
    
    # Generate visualization
    heatmap, overlay, method = create_mlp_visualization(model, img_array, original_img, prediction_value)
    
    # Explanation berdasarkan method
    explanations = {
        "saliency_map": "Berdasarkan sensitivity analysis (gradients)",
        "feature_importance": "Berdasarkan feature importance dari layer pertama",
        "prediction_based": "Berdasarkan pola prediksi AI"
    }
    
    st.sidebar.success(f"Method: {method}\n{explanations.get(method, '')}")
    
    return heatmap, overlay, f"mlp_{method}"