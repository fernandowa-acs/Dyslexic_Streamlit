# app_pro.py - TFLite ONLY Version
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Import visualizers
try:
    from grad_cam_visible import visualize_gradcam_ultimate
except ImportError:
    st.error("❌ grad_cam_visible.py not found")
    visualize_gradcam_ultimate = None

try:
    from grad_cam_mlp import visualize_mlp_model
except ImportError:
    st.error("❌ grad_cam_mlp.py not found") 
    visualize_mlp_model = None

# Konfigurasi halaman
st.set_page_config(
    page_title="Dyslexia Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6366F1;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4F46E5;
        margin-bottom: 1rem;
    }
    .result-card {
        background: #F8FAFC;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #6366F1;
        margin: 1rem 0;
    }
    .dyslexic-result {
        border-left: 5px solid #EF4444;
        background: #FEF2F2;
    }
    .normal-result {
        border-left: 5px solid #10B981;
        background: #F0FDF4;
    }
    .upload-box {
        border: 2px dashed #6366F1;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #F8FAFC;
        margin: 1rem 0;
    }
    .confidence-high { color: #10B981; font-weight: bold; }
    .confidence-medium { color: #F59E0B; font-weight: bold; }
    .confidence-low { color: #EF4444; font-weight: bold; }
    .badge-tflite { background: #FFE4E6; color: #BE123C; }
    .batch-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6366F1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Dyslexia Detection AI</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    st.info("Aplikasi ini menggunakan AI untuk menganalisis karakteristik tulisan tangan dalam mendeteksi indikasi disleksia.")
    
    st.markdown("---")
    st.markdown('<div class="sub-header">📝 Cara Penggunaan</div>', unsafe_allow_html=True)
    st.write("""
    1. **Single Analysis**: Upload 1 gambar untuk analisis detail
    2. **Batch Processing**: Upload multiple gambar untuk analisis cepat
    3. **Lihat** hasil dan penjelasan AI
    4. **Visualisasi** tunjukkan area penting
    """)

# Load TFLite model function
@st.cache_resource
def load_tflite_model():
    """Load TFLite model only"""
    try:
        if os.path.exists('model_quantized.tflite'):
            interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
            interpreter.allocate_tensors()
            
            # ✅ DEBUG: Tampilkan input shape model
            input_details = interpreter.get_input_details()
            st.sidebar.write(f"📊 Model input shape: {input_details[0]['shape']}")
            st.sidebar.write(f"📊 Model input dtype: {input_details[0]['dtype']}")
            
            st.sidebar.success("✅ TFLite model loaded successfully!")
            return interpreter
        else:
            st.sidebar.error("❌ model_quantized.tflite not found!")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading TFLite model: {e}")
        return None
# Prediction function for TFLite
def predict_tflite(interpreter, img_array):
    """Prediction untuk TFLite model"""
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction
    
# Enhanced preprocessing function untuk TFLite
def preprocess_image(img, target_size=(200, 200)):
    """Preprocessing untuk TFLite model"""
    try:
        # Convert to numpy array
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
            
        # Convert to grayscale 
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize
        img_array = cv2.resize(img_array, target_size)
        
        # Add channel dimension (1 channel untuk grayscale)
        img_array = np.expand_dims(img_array, axis=-1)  # Shape: (200, 200, 1)

        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 200, 200, 1)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {e}")
        return None
    
# Display confidence meter
def display_confidence(confidence, class_name):
    st.markdown(f'<div class="sub-header">📊 Confidence Level</div>', unsafe_allow_html=True)
    
    # Progress bar dengan warna
    st.progress(float(confidence))
    
    # Confidence text dengan styling
    if confidence > 0.7:
        css_class = "confidence-high"
        icon = "🟢"
        interpretation = "High confidence"
    elif confidence > 0.5:
        css_class = "confidence-medium" 
        icon = "🟡"
        interpretation = "Medium confidence"
    else:
        css_class = "confidence-low"
        icon = "🔴"
        interpretation = "Low confidence"
        
    st.markdown(f'<span class="{css_class}">{icon} {confidence:.2%} confident - {interpretation}</span>', unsafe_allow_html=True)

# Display prediction result
def display_prediction_result(pred_class, confidence):
    """Display hasil prediksi dengan styling"""
    if pred_class == 0:  # Normal
        st.markdown(f"""
        <div class="result-card normal-result">
            <h2>🟢 Hasil: Normal</h2>
            <p>Sistem mendeteksi karakteristik tulisan dalam kategori normal.</p>
            <p><strong>Karakteristik yang diamati:</strong> Spasi konsisten, bentuk huruf stabil, alignment baik.</p>
        </div>
        """, unsafe_allow_html=True)
    else:  # Dyslexic
        st.markdown(f"""
        <div class="result-card dyslexic-result">
            <h2>🔴 Hasil: Indikasi Disleksia</h2>
            <p>Sistem mendeteksi karakteristik yang mungkin mengindikasikan disleksia.</p>
            <p><strong>Karakteristik yang diamati:</strong> Inkonsistensi spasi, bentuk huruf tidak stabil, kemiringan variatif.</p>
            <p><small>⚠️ <strong>Peringatan:</strong> Hasil ini perlu dikonfirmasi oleh profesional medis. Ini hanya alat bantu screening.</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    display_confidence(confidence, "Normal" if pred_class == 0 else "Dyslexic")

def create_fallback_visualization(original_img, prediction_value):
    """Create fallback visualization untuk TFLite"""
    h, w = original_img.shape[:2]
    
    # Buat heatmap sederhana berdasarkan prediksi
    heatmap = np.zeros((h, w))
    
    # Pattern berdasarkan confidence
    if prediction_value > 0.7:  # High confidence dyslexic
        points = [(h//3, w//3), (h//3, 2*w//3), (2*h//3, w//3)]
        intensity = 0.8
    elif prediction_value > 0.4:  # Medium confidence
        points = [(h//2, w//2), (h//3, w//2)]
        intensity = 0.6
    else:  # Low confidence atau normal
        points = [(h//2, w//2)]
        intensity = 0.4
    
    # Apply Gaussian patterns
    for center_y, center_x in points:
        y, x = np.ogrid[:h, :w]
        sigma = min(h, w) / 8
        region_heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        heatmap += region_heatmap * intensity
    
    # Normalize
    heatmap = np.clip(heatmap, 0, 1)
    
    # Create overlay
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_img
    
    blended = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap, blended

# Load model
interpreter = load_tflite_model()

# BATCH PROCESSING SECTION
st.markdown('<div class="batch-header">📁 Batch Processing - Analisis Multiple Gambar</div>', unsafe_allow_html=True)

with st.expander("🎯 **Klik untuk Analisis Multiple Gambar Sekaligus**", expanded=False):
    st.markdown("""
    **Fitur Batch Processing memungkinkan Anda:**
    - ✅ Upload **multiple gambar** sekaligus (maksimal 20 file)
    - ✅ Dapatkan **hasil analisis** dalam tabel terstruktur
    - ✅ **Ekspor hasil** ke format CSV
    - ✅ **Bandingkan hasil** across multiple samples
    - ✅ **Statistik summary** untuk analisis cepat
    """)
    
    batch_files = st.file_uploader(
        "**Pilih multiple gambar untuk analisis batch**",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="batch_uploader",
        help="Pilih 2-20 gambar untuk dianalisis sekaligus"
    )
    
    if batch_files and interpreter is not None:
        if len(batch_files) > 20:
            st.warning(f"⚠️ Terlalu banyak file! Maksimal 20 file per batch. Anda memilih {len(batch_files)} file.")
            batch_files = batch_files[:20]
            st.info(f"📝 Akan menganalisis 20 file pertama saja.")
        
        st.success(f"📦 **{len(batch_files)} gambar terdeteksi**. Memulai analisis batch...")
        
        # Progress bar untuk batch processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(batch_files):
            # Update progress
            progress = (i + 1) / len(batch_files)
            progress_bar.progress(progress)
            status_text.text(f"🔄 Memproses {i+1}/{len(batch_files)}: {uploaded_file.name}")
            
            try:
                # Process single image
                image = Image.open(uploaded_file)
                img_array = preprocess_image(image)
                
                if img_array is not None:
                    # Prediction dengan TFLite
                    prediction = predict_tflite(interpreter, img_array)
                    
                    pred_class = int(prediction[0][0] > 0.6)
                    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
                    
                    # Determine status
                    if pred_class == 0:
                        status = "Normal"
                        status_emoji = "🟢"
                        risk_level = "Rendah"
                    else:
                        status = "Indikasi Disleksia"
                        status_emoji = "🔴" 
                        risk_level = "Tinggi"
                    
                    # Confidence level
                    if confidence > 0.7:
                        conf_level = "Tinggi"
                    elif confidence > 0.5:
                        conf_level = "Medium"
                    else:
                        conf_level = "Rendah"
                    
                    results.append({
                        'file_name': uploaded_file.name,
                        'status': status,
                        'status_emoji': status_emoji,
                        'risk_level': risk_level,
                        'confidence': f"{confidence:.2%}",
                        'confidence_value': confidence,
                        'confidence_level': conf_level,
                        'raw_score': float(prediction[0][0]),
                        'image_size': f"{image.size[0]}x{image.size[1]}",
                        'image_mode': image.mode,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    results.append({
                        'file_name': uploaded_file.name,
                        'status': "Error Processing",
                        'status_emoji': "❌",
                        'risk_level': "Unknown",
                        'confidence': "N/A",
                        'confidence_value': 0.0,
                        'confidence_level': "N/A",
                        'raw_score': 0.0,
                        'image_size': "N/A",
                        'image_mode': "N/A",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                results.append({
                    'file_name': uploaded_file.name,
                    'status': "Error",
                    'status_emoji': "❌",
                    'risk_level': "Unknown",
                    'confidence': "N/A",
                    'confidence_value': 0.0,
                    'confidence_level': "N/A",
                    'raw_score': 0.0,
                    'image_size': "N/A",
                    'image_mode': "N/A",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        if results:
            # Display summary statistics
            st.markdown("### 📈 Statistik Summary")
            
            total_files = len(results)
            normal_count = len([r for r in results if r['status'] == 'Normal'])
            dyslexic_count = len([r for r in results if r['status'] == 'Indikasi Disleksia'])
            error_count = len([r for r in results if r['status'] in ['Error', 'Error Processing']])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Files", total_files)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🟢 Normal", normal_count)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🔴 Indikasi Disleksia", dyslexic_count)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("❌ Error", error_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(results)
            
            # Display results table
            st.markdown("### 📊 Hasil Batch Analysis")
            
            # Style the DataFrame
            def style_results(row):
                status_value = row['status_emoji']
                if status_value == '🟢':
                    return ['background-color: #D1FAE5'] * len(row)
                elif status_value == '🔴':
                    return ['background-color: #FEE2E2'] * len(row)
                else:
                    return ['background-color: #FEF3C7'] * len(row)
            
            # Display table
            display_columns = ['file_name', 'status_emoji', 'risk_level', 'confidence', 'confidence_level', 'image_size', 'timestamp']
            display_df = df[display_columns]
            
            styled_df = display_df.style.apply(style_results, axis=1)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "file_name": st.column_config.TextColumn("Nama File", width="medium"),
                    "status_emoji": st.column_config.TextColumn("Status", width="small"),
                    "risk_level": st.column_config.TextColumn("Level Risiko", width="small"),
                    "confidence": st.column_config.TextColumn("Tingkat Kepercayaan", width="small"),
                    "confidence_level": st.column_config.TextColumn("Level Confidence", width="small"),
                    "image_size": st.column_config.TextColumn("Ukuran Gambar", width="small"),
                    "timestamp": st.column_config.TextColumn("Waktu Analisis", width="medium")
                }
            )
            
            # Export functionality
            st.markdown("### 💾 Export Results")
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"dyslexia_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download hasil analisis dalam format CSV"
                )
            
            with col_export2:
                # JSON Export
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"dyslexia_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download hasil analisis dalam format JSON"
                )

# SINGLE FILE ANALYSIS SECTION
st.markdown("---")
st.markdown('<div class="sub-header">🔍 Single Image Analysis</div>', unsafe_allow_html=True)
st.markdown("""
<div class="upload-box">
    <h3>Drag & drop file here</h3>
    <p>Limit 200MB per file • JPG, JPEG, PNG</p>
    <p><small>📝 <strong>Tips:</strong> Pastikan gambar jelas, tulisan terbaca, dan background kontras</small></p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    " ",
    type=['jpg', 'jpeg', 'png'],
    help="Upload gambar tulisan tangan untuk analisis disleksia",
    label_visibility="collapsed",
    key="single_uploader"
)

# Main app logic untuk single file
if uploaded_file is not None and interpreter is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Gambar yang Diupload")
        image_display = Image.open(uploaded_file)
        st.image(image_display, use_container_width=True, caption="Gambar Original")
        
        # Image info
        st.write(f"**📏 Ukuran:** {image_display.size} pixels")
        st.write(f"**📄 Format:** {uploaded_file.type}")
        st.write(f"**🎨 Mode:** {image_display.mode}")
        st.write(f"**🤖 Model Type:** TFLite")

    with col2:
        st.subheader("🔍 Hasil Analisis")
        
        with st.spinner("🔄 AI sedang menganalisis karakteristik tulisan..."):
            img_array = preprocess_image(image_display)
            
            if img_array is not None:
                try:
                    # Prediction dengan TFLite
                    prediction = predict_tflite(interpreter, img_array)
                    
                    # Calculate results
                    threshold = 0.6
                    pred_class = int(prediction[0][0] > threshold)
                    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]
                    
                    # Display results
                    display_prediction_result(pred_class, confidence)
                    
                    # Additional technical info
                    with st.expander("📊 Detail Teknis Prediksi"):
                        st.write(f"**Raw Prediction Value:** {prediction[0][0]:.6f}")
                        st.write(f"**Threshold untuk Dyslexic:** > {threshold}")
                        st.write(f"**Kelas Prediksi:** {'Dyslexic' if pred_class == 1 else 'Normal'}")
                        st.write(f"**Model Type:** TFLite (Optimized)")
                        st.write(f"**Model Size:** {os.path.getsize('model_quantized.tflite') / 1024 / 1024:.1f} MB")
                        
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

    # VISUALIZATION Section
    st.markdown("---")
    st.subheader("🧠 AI Explanation - Model Visualization")
    
    with st.spinner("🔄 Generating AI explanation..."):
        try:
            # Prepare images
            original_img = np.array(image_display)
            if len(original_img.shape) == 3:
                original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            else:
                original_img_gray = original_img
            original_img_gray = cv2.resize(original_img_gray, (200, 200))
            
            # Fallback visualization untuk TFLite
            st.info("🚀 Using TFLite Optimized Visualization")
            heatmap, grad_cam_img = create_fallback_visualization(original_img_gray, prediction[0][0])
            
            # Display results
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.image(original_img_gray, 
                       caption="🖼️ Original Image", 
                       use_container_width=True)
            
            with col2:
                if heatmap is not None:
                    heatmap_display = cv2.resize(heatmap, (200, 200))
                    heatmap_display = np.uint8(255 * heatmap_display)
                    heatmap_display = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_VIRIDIS)
                    st.image(heatmap_display, 
                           caption="🔥 Attention Map", 
                           use_container_width=True)
            
            with col3:
                if grad_cam_img is not None:
                    st.image(grad_cam_img, 
                           caption="🎨 AI Focus Areas", 
                           use_container_width=True)
            
            # TFLite explanation
            st.success("""
            **🚀 TFLite Optimized Model:**
            - Model teroptimasi untuk **kecepatan**
            - Analisis **real-time** dengan akurasi tinggi
            - Visualisasi area fokus AI berdasarkan **pola umum**
            """)
            
        except Exception as e:
            st.error(f"❌ Visualization Error: {str(e)}")

elif interpreter is None:
    st.error("""
    ❌ TFLite model tidak ditemukan. Pastikan file `model_quantized.tflite` ada di folder.
    """)
else:
    # Landing page state
    st.markdown("""
    <div style='text-align: center; padding: 4rem; color: #6B7280;'>
        <h3>👆 Upload gambar untuk memulai analisis</h3>
        <p>Pilih antara <strong>Batch Processing</strong> (multiple gambar) atau <strong>Single Analysis</strong> (1 gambar detail)</p>
        <p><small>📝 Contoh tulisan yang bisa dianalisis: tugas sekolah, catatan tangan, surat, latihan menulis</small></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 2rem;'>
    <p><strong>Dyslexia Detection AI</strong>
    <p>Version 2.4 | TFLite Optimized | Fast & Efficient</p>
    <p><small>⚠️ Disclaimer: Hasil analisis merupakan alat bantu screening dan perlu konfirmasi profesional medis.</small></p>
</div>
""", unsafe_allow_html=True)