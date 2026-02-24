import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from config import *

class FlexibleStressModel:
    """
    Flexible stress/emotion detection model that can adapt to any number of emotion classes
    using transfer learning and universal encoding.
    """
    
    def __init__(self, base_model_path=None):
        self.base_model_path = base_model_path or os.path.join(BASE_DIR, "stress_model.keras")
        self.model = None
        self.encoder = None
        self.emotion_mapping = {}
        self.stress_mapping = {}
        self.num_classes = 0
        self.input_shape = (130, N_MFCC)  # (time_steps, features)
        self.scaler = None
        
    def load_base_model(self):
        """Load the pre-trained base model for feature extraction"""
        try:
            base_model = load_model(self.base_model_path)
            print(f"Loaded base model from {self.base_model_path}")
            
            # Ensure the base model is built
            base_model.build(input_shape=(None, 130, 40))
            
            # Remove the final dense layer to get feature extractor
            # Create new model that outputs features from the layer before final classification
            feature_extractor = Model(
                inputs=base_model.input,
                outputs=base_model.layers[-2].output,  # Layer before final softmax
                name="feature_extractor"
            )
            
            # Partially unfreeze the base layers for transfer learning
            # The base model contains Conv1D and LSTM layers. We want to unfreeze the last few LSTM layers
            for layer in feature_extractor.layers:
                if 'lstm' in layer.name.lower():
                    layer.trainable = True
                else:
                    layer.trainable = False
                
            return feature_extractor
        except Exception as e:
            print(f"Error loading base model: {e}")
            return None
    
    def create_adaptive_model(self, emotion_classes):
        """
        Create a model adapted to specific emotion classes using transfer learning
        """
        self.num_classes = len(emotion_classes)
        
        # Load base feature extractor
        feature_extractor = self.load_base_model()
        if feature_extractor is None:
            # If no base model, create from scratch
            return self._create_model_from_scratch(emotion_classes)
        
        # Create new adaptive model
        input_layer = Input(shape=self.input_shape)
        
        # Use frozen feature extractor
        features = feature_extractor(input_layer, training=False)
        
        # Add new trainable layers for adaptation
        x = Dense(128, activation='relu')(features)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        # Final output layer with correct number of classes
        output_layer = Dense(self.num_classes, activation='softmax', name='emotion_output')(x)
        
        # Create the complete model
        model = Model(inputs=input_layer, outputs=output_layer, name=f"adaptive_emotion_model_{self.num_classes}classes")
        
        # Compile with appropriate loss and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"Created adaptive model for {self.num_classes} emotion classes")
        return model
    
    def _create_model_from_scratch(self, emotion_classes):
        """Create a new model from scratch if no base model is available"""
        self.num_classes = len(emotion_classes)
        
        model = tf.keras.Sequential([
            Input(shape=self.input_shape),
            Conv1D(256, kernel_size=8, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=8, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=8, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def setup_emotion_mappings(self, emotion_classes):
        """Setup emotion to stress mapping and save configurations"""
        # Default stress mapping (can be customized)
        stress_emotions = ['angry', 'fearful', 'disgust', 'sad']
        neutral_emotions = ['neutral', 'calm']
        positive_emotions = ['happy', 'surprised', 'pleasant_surprise', 'pleasant_surprised']
        
        self.emotion_mapping = {i: emotion for i, emotion in enumerate(emotion_classes)}
        
        # Create stress mapping
        self.stress_mapping = {}
        for i, emotion in enumerate(emotion_classes):
            emotion_lower = emotion.lower()
            if any(se in emotion_lower for se in stress_emotions):
                self.stress_mapping[i] = 'Stressed'
            elif any(ne in emotion_lower for ne in neutral_emotions):
                self.stress_mapping[i] = 'Not Stressed'
            elif any(pe in emotion_lower for pe in positive_emotions):
                self.stress_mapping[i] = 'Not Stressed'
            else:
                # Default to not stressed for unknown emotions
                self.stress_mapping[i] = 'Not Stressed'
        
        print("Emotion mappings:")
        for i, emotion in self.emotion_mapping.items():
            print(f"  {i}: {emotion} -> {self.stress_mapping[i]}")
    
    def train_adaptive_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, class_weight=None):
        """Train the adaptive model on new dataset"""
        if self.model is None:
            raise ValueError("Model not created. Call create_adaptive_model() first.")
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                f'adaptive_model_{self.num_classes}classes.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        print(f"Training adaptive model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2
        )
        
        return history
    
    def predict_emotion(self, features):
        """Predict emotion and return both emotion and stress status"""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Ensure features have correct shape
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
            
        # Apply standard scaling if scaler was loaded
        if self.scaler is not None:
            N, T, F = features.shape
            features_flat = features.reshape(N, T * F)
            features_flat = self.scaler.transform(features_flat)
            features = features_flat.reshape(N, T, F)
            
        
        # Get predictions
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get top emotion
        emotion_idx = np.argmax(predictions)
        emotion = self.emotion_mapping[emotion_idx]
        confidence = float(predictions[emotion_idx])
        
        # Get stress status
        stress_status = self.stress_mapping[emotion_idx]
        
        return {
            'emotion': emotion,
            'emotion_index': emotion_idx,
            'confidence': confidence,
            'stress_status': stress_status,
            'all_probabilities': {self.emotion_mapping[i]: float(p) for i, p in enumerate(predictions)}
        }
    
    def save_model(self, save_path):
        """Save the adaptive model and configurations"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Save the model
        model_path = f"{save_path}_model.keras"
        self.model.save(model_path)
        
        # Save configurations
        config = {
            'emotion_mapping': self.emotion_mapping,
            'stress_mapping': self.stress_mapping,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }
        
        config_path = f"{save_path}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")
        
        return model_path, config_path
    
    def load_model(self, model_path, config_path):
        """Load a saved adaptive model with configurations"""
        # Load model
        self.model = load_model(model_path)
        
        # Load configurations
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.emotion_mapping = {int(k): v for k, v in config['emotion_mapping'].items()}
        self.stress_mapping = {int(k): v for k, v in config['stress_mapping'].items()}
        self.num_classes = config['num_classes']
        self.input_shape = tuple(config['input_shape'])
        
        print(f"Loaded model with {self.num_classes} emotion classes")
        
        # Load scaler if it exists alongside the model
        model_dir = os.path.dirname(model_path)
        scaler_path = os.path.join(model_dir, "universal_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                print(f"Could not load scaler: {e}")
                
        return True
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Detailed classification report
        class_names = [self.emotion_mapping[i] for i in range(self.num_classes)]
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=class_names, output_dict=True)
        
        # Stress-level accuracy
        stress_true = [self.stress_mapping[i] for i in y_true_classes]
        stress_pred = [self.stress_mapping[i] for i in y_pred_classes]
        stress_accuracy = accuracy_score(stress_true, stress_pred)
        
        results = {
            'emotion_accuracy': accuracy,
            'stress_accuracy': stress_accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes),
            'class_names': class_names
        }
        
        return results
