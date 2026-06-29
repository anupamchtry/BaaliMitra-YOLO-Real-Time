"""
Crop Disease Detection - Flask Backend
Real-time detection with cascaded YOLO models
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import json
import time

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATHS = {
    'stage1': 'models/stage1_balanced_leaf_detector.pt',
    'stage2': 'models/stage2_species_classifier.pt',
    'stage3': {
        'potato': 'models/stage3_potato_disease.pt',
        'tomato': 'models/stage3_tomato_disease.pt',
        'rice': 'models/stage3_rice_disease.pt'
    }
}

CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# LOAD MODELS
# ============================================================================

class CropDiseaseDetector:
    def __init__(self):
        print("🔄 Loading models...")
        self.models_status = {
            'stage1': False,
            'stage2': False,
            'stage3': {}
        }
        
        try:
            self.stage1 = YOLO(MODEL_PATHS['stage1'])
            self.models_status['stage1'] = True
            print("✅ Stage 1 loaded (Leaf Detector)")
        except FileNotFoundError as e:
            print(f"❌ Stage 1 failed - Model file not found: {e}")
            self.stage1 = None
        except Exception as e:
            print(f"❌ Stage 1 failed - Error: {e}")
            self.stage1 = None
        
        try:
            self.stage2 = YOLO(MODEL_PATHS['stage2'])
            self.models_status['stage2'] = True
            print("✅ Stage 2 loaded (Species Classifier)")
        except FileNotFoundError as e:
            print(f"❌ Stage 2 failed - Model file not found: {e}")
            self.stage2 = None
        except Exception as e:
            print(f"❌ Stage 2 failed - Error: {e}")
            self.stage2 = None
        
        self.stage3 = {}
        for plant, path in MODEL_PATHS['stage3'].items():
            try:
                self.stage3[plant] = YOLO(path)
                self.models_status['stage3'][plant] = True
                print(f"✅ Stage 3 loaded ({plant.title()} Disease)")
            except FileNotFoundError as e:
                print(f"❌ Stage 3 ({plant}) failed - Model file not found: {e}")
                self.models_status['stage3'][plant] = False
            except Exception as e:
                print(f"❌ Stage 3 ({plant}) failed - Error: {e}")
                self.models_status['stage3'][plant] = False
        
        # Validate critical models loaded
        if not self.models_status['stage1']:
            print("⚠️  WARNING: Stage 1 model is required but failed to load!")
        
        print("✅ Model loading complete!\n")
    
    def predict_frame(self, frame, conf_threshold=CONFIDENCE_THRESHOLD):
        """
        Run cascaded detection on a frame
        Returns only detection data (drawing handled by frontend)
        
        Args:
            frame: Input frame/image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            detections: List of detection results
        """
        detections = []
        
        # Validate frame
        if frame is None or frame.size == 0:
            print("❌ Invalid frame: Empty or None")
            return detections
        
        # Validate models
        if self.stage1 is None:
            print("❌ Cannot predict: Stage 1 model not loaded")
            return detections
        
        try:
            # Stage 1: Detect objects
            results1 = self.stage1.predict(frame, conf=conf_threshold, verbose=False)
            
            if not results1 or len(results1) == 0:
                return detections
            
            for result in results1[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(cls)
                
                class_name = self.stage1.names[cls]
                is_leaf = 'leaf' in class_name.lower() and 'non' not in class_name.lower()
                
                if not is_leaf:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'type': 'non_leaf',
                        'confidence': float(conf)
                    })
                    continue
                
                # Crop leaf region
                leaf_crop = frame[y1:y2, x1:x2]
                
                if leaf_crop.size == 0:
                    continue
                
                # Stage 2: Classify species
                if self.stage2 is None:
                    continue
                
                try:
                    results2 = self.stage2.predict(leaf_crop, conf=conf_threshold, verbose=False)
                    
                    if len(results2[0].boxes) == 0:
                        continue
                    
                    species_box = results2[0].boxes.data[0]
                    species_conf = float(species_box[4])
                    species_cls = int(species_box[5])
                    species_name = self.stage2.names[species_cls].lower()
                except Exception as e:
                    print(f"⚠️  Stage 2 error: {e}")
                    continue
                
                # Stage 3: Detect disease
                if species_name not in self.stage3:
                    continue
                
                try:
                    results3 = self.stage3[species_name].predict(leaf_crop, conf=conf_threshold, verbose=False)
                    
                    if len(results3[0].boxes) > 0:
                        disease_box = results3[0].boxes.data[0]
                        disease_conf = float(disease_box[4])
                        disease_cls = int(disease_box[5])
                        disease_name = self.stage3[species_name].names[disease_cls]
                        
                        is_healthy = 'healthy' in disease_name.lower()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'type': 'leaf',
                            'species': species_name,
                            'species_confidence': species_conf,
                            'disease': disease_name,
                            'disease_confidence': disease_conf,
                            'is_healthy': is_healthy
                        })
                except Exception as e:
                    print(f"⚠️  Stage 3 error for {species_name}: {e}")
                    continue
            
            return detections
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return detections


# Initialize detector
print("\n" + "="*70)
print("🌿 CROP DISEASE DETECTION SERVER")
print("="*70 + "\n")

detector = CropDiseaseDetector()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint with model status"""
    return jsonify({
        'status': 'healthy' if detector.models_status['stage1'] else 'degraded',
        'models_status': detector.models_status,
        'models_loaded': {
            'stage1': detector.stage1 is not None,
            'stage2': detector.stage2 is not None,
            'stage3_potato': 'potato' in detector.stage3,
            'stage3_tomato': 'tomato' in detector.stage3,
            'stage3_rice': 'rice' in detector.stage3
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Process uploaded image or webcam frame"""
    try:
        # Validate request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Validate model is loaded
        if not detector.models_status['stage1']:
            return jsonify({
                'success': False,
                'error': 'Stage 1 model not loaded. Server is not ready.'
            }), 503
        
        try:
            # Decode base64 image
            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image format: {str(e)}'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image decoding error: {str(e)}'
            }), 400
        
        # Run detection
        start_time = time.time()
        detections = detector.predict_frame(frame, conf_threshold=CONFIDENCE_THRESHOLD)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return jsonify({
            'success': True,
            'detections': detections,
            'inference_time': round(inference_time, 2),
            'count': len(detections)
        })
    
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/models/info')
def models_info():
    """Get model information"""
    info = {
        'stage1': {
            'name': 'Leaf Detector',
            'classes': list(detector.stage1.names.values()) if detector.stage1 else [],
            'loaded': detector.stage1 is not None
        },
        'stage2': {
            'name': 'Species Classifier',
            'classes': list(detector.stage2.names.values()) if detector.stage2 else [],
            'loaded': detector.stage2 is not None
        },
        'stage3': {}
    }
    
    for plant, model in detector.stage3.items():
        info['stage3'][plant] = {
            'name': f'{plant.title()} Disease Detector',
            'classes': list(model.names.values()),
            'loaded': True
        }
    
    return jsonify(info)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)