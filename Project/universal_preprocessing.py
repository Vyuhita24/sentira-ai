import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import json
from collections import defaultdict
import re
from config import *

class UniversalPreprocessor:
    """
    Universal preprocessor that can handle any emotion dataset with flexible emotion detection
    and consistent encoding across different datasets.
    """
    
    def __init__(self):
        self.universal_encoder = None
        self.emotion_stats = defaultdict(int)
        self.dataset_emotions = set()
        
    def detect_emotion_from_path(self, file_path):
        """
        Advanced emotion detection from file paths and folder names.
        Works with RAVDESS, TESS, and custom datasets.
        """
        file_name = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        # Convert to lowercase for matching
        folder_lower = folder_name.lower()
        file_lower = file_name.lower()
        parent_lower = parent_folder.lower()
        
        # RAVDESS format: 03-01-06-01-02-01-12.wav (emotion code is 3rd part)
        ravdess_match = re.match(r'\d{2}-\d{2}-(\d{2})-', file_name)
        if ravdess_match:
            emotion_code = ravdess_match.group(1)
            ravdess_emotions = {
                "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
            }
            if emotion_code in ravdess_emotions:
                return ravdess_emotions[emotion_code]
        
        # TESS format: folder names contain emotions
        tess_emotions = {
            'angry': 'angry',
            'disgust': 'disgust', 
            'fear': 'fearful',
            'fearful': 'fearful',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'surprise': 'surprised',
            'surprised': 'surprised',
            'pleasant_surprise': 'pleasant_surprise',
            'pleasant_surprised': 'pleasant_surprised',
            'pleasant': 'pleasant',
            'calm': 'calm',
            'excited': 'excited'
        }
        
        # Check folder name first
        for key, emotion in tess_emotions.items():
            if key in folder_lower:
                return emotion
        
        # Check parent folder (for nested structures)
        for key, emotion in tess_emotions.items():
            if key in parent_lower:
                return emotion
        
        # Check filename directly
        for key, emotion in tess_emotions.items():
            if key in file_lower:
                return emotion
        
        # Custom pattern matching for other datasets
        # Look for emotion keywords in the full path
        full_path_lower = (folder_name + " " + file_name).lower()
        
        emotion_patterns = {
            'angry': r'\b(angry|mad|furious|rage|annoyed)\b',
            'sad': r'\b(sad|down|unhappy|depressed|cry|tears|miserable)\b',
            'happy': r'\b(happy|joy|delight|pleased|excited|glad|cheerful|elated)\b',
            'fearful': r'\b(afraid|scared|fear|terrified|nervous|anxious|panic)\b',
            'disgust': r'\b(disgust|disgusted|gross|nausea|revulsion)\b',
            'surprised': r'\b(surprised|shocked|amazed|astonished|startled)\b',
            'neutral': r'\b(neutral|normal|average|calm|peaceful)\b',
            'pleasant': r'\b(pleasant|lovely|beautiful|wonderful|nice)\b'
        }
        
        for emotion, pattern in emotion_patterns.items():
            if re.search(pattern, full_path_lower):
                return emotion
        
        # If no emotion detected, return None
        return None
    
    def extract_features(self, file_path, use_augmentation=True, num_inference_windows=1):
        """
        Extract MFCC features with data augmentation.
        Returns a list of feature arrays (original + augmented).
        """
        try:
            # Load full audio file first. For live recordings, speech may start late.
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Trim leading/trailing silence to reduce bias from recording latency/noise.
            trimmed_y, _ = librosa.effects.trim(y, top_db=25)
            if len(trimmed_y) > 0:
                y = trimmed_y
            
            segments = []

            # In inference mode, sample multiple high-energy windows to stabilize predictions.
            if not use_augmentation and num_inference_windows > 1 and len(y) > SAMPLES_PER_TRACK:
                max_start = len(y) - SAMPLES_PER_TRACK
                scan_step = max(1, SAMPLES_PER_TRACK // 6)
                candidates = []

                for start in range(0, max_start + 1, scan_step):
                    segment = y[start:start + SAMPLES_PER_TRACK]
                    energy = float(np.mean(segment ** 2))
                    candidates.append((energy, start))

                candidates.sort(reverse=True, key=lambda x: x[0])

                min_gap = max(1, SAMPLES_PER_TRACK // 4)
                selected_starts = []
                for _, start in candidates:
                    if all(abs(start - existing) >= min_gap for existing in selected_starts):
                        selected_starts.append(start)
                    if len(selected_starts) >= num_inference_windows:
                        break

                if not selected_starts:
                    selected_starts = [0]

                for start in selected_starts:
                    segments.append(y[start:start + SAMPLES_PER_TRACK])
            else:
                # Use the highest-energy segment if audio is longer than target length.
                # This keeps the most informative speech chunk instead of arbitrary leading audio.
                if len(y) > SAMPLES_PER_TRACK:
                    step = max(1, SAMPLES_PER_TRACK // 8)
                    max_energy = -1.0
                    best_start = 0
                    max_start = len(y) - SAMPLES_PER_TRACK

                    for start in range(0, max_start + 1, step):
                        segment = y[start:start + SAMPLES_PER_TRACK]
                        energy = float(np.mean(segment ** 2))
                        if energy > max_energy:
                            max_energy = energy
                            best_start = start

                    y = y[best_start:best_start + SAMPLES_PER_TRACK]
                else:
                    padding = SAMPLES_PER_TRACK - len(y)
                    y = np.pad(y, (0, padding), mode='constant')
                segments = [y]
            
            # Data augmentation functions
            def augment_audio(audio, sample_rate):
                # Original
                yield audio
                
                if use_augmentation:
                    # Add small noise
                    noise = np.random.normal(0, 0.005, audio.shape)
                    yield audio + noise
                    
                    # Pitch shift
                    yield librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)
                    
                    # Time stretch (slower)
                    yield librosa.effects.time_stretch(audio, rate=0.9)
                    
                    # Time stretch (faster)
                    yield librosa.effects.time_stretch(audio, rate=1.1)
            
            mfccs_list = []
            FIXED_FRAMES = 130  # Fixed number of time frames
            
            for segment in segments:
                for aug_y in augment_audio(segment, sr):
                    mfccs = librosa.feature.mfcc(
                        y=aug_y, 
                        sr=sr, 
                        n_mfcc=N_MFCC, 
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH
                    )
                    mfccs = mfccs.T  # Shape: (frames, n_mfcc)
                    
                    # Pad or truncate to fixed frames
                    if mfccs.shape[0] > FIXED_FRAMES:
                        mfccs = mfccs[:FIXED_FRAMES, :]
                    elif mfccs.shape[0] < FIXED_FRAMES:
                        pad_width = FIXED_FRAMES - mfccs.shape[0]
                        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
                    
                    mfccs_list.append(mfccs)
            
            return mfccs_list
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def scan_dataset(self, dataset_dir):
        """
        Scan dataset to identify all emotions and their counts.
        Returns a dictionary of emotion statistics.
        """
        print(f"Scanning dataset at: {dataset_dir}")
        
        emotion_counts = defaultdict(int)
        total_files = 0
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        all_files = []
        for ext in audio_extensions:
            all_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
        
        print(f"Found {len(all_files)} audio files")
        
        for file_path in all_files:
            emotion = self.detect_emotion_from_path(file_path)
            if emotion:
                emotion_counts[emotion] += 1
                total_files += 1
                self.dataset_emotions.add(emotion)
            else:
                print(f"Warning: Could not detect emotion for {file_path}")
        
        print(f"\nDataset Statistics:")
        print(f"Total files: {total_files}")
        print(f"Unique emotions: {len(emotion_counts)}")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count} files")
        
        return dict(emotion_counts)
    
    def load_or_create_encoder(self, emotion_classes, encoder_path=None):
        """
        Load existing universal encoder or create new one.
        Ensures consistent encoding across different datasets.
        """
        if encoder_path and os.path.exists(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                
                # Check if all current emotions are in the encoder
                existing_classes = encoder.categories_[0].tolist()
                missing_emotions = set(emotion_classes) - set(existing_classes)
                
                if missing_emotions:
                    print(f"Warning: New emotions detected: {missing_emotions}")
                    print("Creating new encoder with all emotions...")
                    
                    # Create new encoder with all emotions
                    all_emotions = sorted(list(set(existing_classes) | set(emotion_classes)))
                    new_encoder = OneHotEncoder(sparse_output=False)
                    new_encoder.fit(np.array(all_emotions).reshape(-1, 1))
                    
                    # Save the updated encoder
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(new_encoder, f)
                    
                    return new_encoder
                else:
                    print("Using existing universal encoder")
                    return encoder
                    
            except Exception as e:
                print(f"Error loading encoder: {e}")
        
        # Create new encoder
        print("Creating new universal encoder")
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(np.array(emotion_classes).reshape(-1, 1))
        
        # Save if path provided
        if encoder_path:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
            print(f"Saved universal encoder to {encoder_path}")
        
        return encoder
    
    def process_dataset(self, dataset_dir, output_dir, val_split=0.2, use_augmentation=True):
        """
        Process any emotion dataset and save standardized features.
        """
        print(f"Processing dataset: {dataset_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Scan dataset first
        emotion_stats = self.scan_dataset(dataset_dir)
        emotion_classes = list(emotion_stats.keys())
        
        if not emotion_classes:
            raise ValueError("No emotions detected in dataset!")
        
        # Setup universal encoder
        encoder_path = os.path.join(BASE_DIR, "data", "universal_encoder.pkl")
        self.universal_encoder = self.load_or_create_encoder(emotion_classes, encoder_path)
        
        # Load all data
        X, y = [], []
        processed_files = 0
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
        all_files = []
        for ext in audio_extensions:
            all_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
        
        print(f"Processing {len(all_files)} files...")
        
        for file_path in all_files:
            emotion = self.detect_emotion_from_path(file_path)
            if not emotion:
                continue
            
            features_list = self.extract_features(file_path)
            if features_list is None:
                continue
            
            # Add features and labels
            for features in features_list:
                X.append(features)
                y.append(emotion)
            
            processed_files += 1
            if processed_files % 100 == 0:
                print(f"Processed {processed_files}/{len(all_files)} files...")
        
        if not X:
            raise ValueError("No valid features extracted!")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Encode labels
        y_encoded = self.universal_encoder.transform(y.reshape(-1, 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=val_split, random_state=42, stratify=y_encoded
        )
        
        # Save processed data
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # Save encoder and metadata
        with open(os.path.join(output_dir, "encoder.pkl"), "wb") as f:
            pickle.dump(self.universal_encoder, f)
        
        # Save dataset metadata
        metadata = {
            'emotion_classes': emotion_classes,
            'emotion_counts': emotion_stats,
            'num_classes': len(emotion_classes),
            'total_files': len(all_files),
            'feature_shape': X.shape[1:],
            'val_split': val_split
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset processing complete!")
        print(f"Saved to: {output_dir}")
        print(f"Emotion classes: {emotion_classes}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, emotion_classes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal preprocessor for any emotion dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    preprocessor = UniversalPreprocessor()
    preprocessor.process_dataset(args.dataset_dir, args.output_dir, args.val_split)
