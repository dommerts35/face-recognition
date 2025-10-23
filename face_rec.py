import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from collections import Counter
from sklearn.model_selection import train_test_split

# Configurar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class SimpleEmotionRecognizer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.model = None
        self.model_file = "simple_emotion_model.pkl"
        self.training_data_file = "simple_training_data.pkl"
        self.training_data = self.load_training_data()
        self.load_or_train_model()

    def load_training_data(self):
        """Cargar datos de entrenamiento existentes"""
        if os.path.exists(self.training_data_file):
            try:
                with open(self.training_data_file, 'rb') as f:
                    data = pickle.load(f)
                    print(f"✅ Datos cargados: {len(data['X'])} muestras")
                    if data['y']:
                        counter = Counter(data['y'])
                        print("📊 Distribución actual:")
                        for emotion, count in counter.items():
                            print(f"  {emotion}: {count} muestras")
                    return data
            except Exception as e:
                print(f"❌ Error cargando datos: {e}")
        return {"X": [], "y": []}

    def save_training_data(self):
        """Guardar datos de entrenamiento"""
        try:
            with open(self.training_data_file, 'wb') as f:
                pickle.dump(self.training_data, f)
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")

    def extract_robust_features(self, landmarks):
        """Extraer características robustas y estables"""
        features = []

        try:
            # Puntos clave esenciales
            key_points = {
                'left_eye_top': landmarks[159],
                'left_eye_bottom': landmarks[145],
                'right_eye_top': landmarks[386],
                'right_eye_bottom': landmarks[374],
                'upper_lip': landmarks[13],
                'lower_lip': landmarks[14],
                'mouth_left': landmarks[61],
                'mouth_right': landmarks[291],
                'left_eyebrow': landmarks[70],
                'right_eyebrow': landmarks[300],
                'forehead': landmarks[10],
            }

            # 1. Características de ojos (simplificadas)
            left_eye_height = abs(key_points['left_eye_top'].y - key_points['left_eye_bottom'].y)
            right_eye_height = abs(key_points['right_eye_top'].y - key_points['right_eye_bottom'].y)
            avg_eye_openness = (left_eye_height + right_eye_height) / 2

            # 2. Características de boca (simplificadas)
            mouth_height = abs(key_points['upper_lip'].y - key_points['lower_lip'].y)
            mouth_width = abs(key_points['mouth_left'].x - key_points['mouth_right'].x)

            # 3. Características de cejas (simplificadas)
            left_eyebrow_height = abs(key_points['left_eyebrow'].y - key_points['forehead'].y)
            right_eyebrow_height = abs(key_points['right_eyebrow'].y - key_points['forehead'].y)
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2

            # 4. Ratios clave
            mouth_eye_ratio = mouth_height / (avg_eye_openness + 1e-6)
            mouth_aspect_ratio = mouth_width / (mouth_height + 1e-6)

            # 5. Asimetrías (simplificadas)
            eye_asymmetry = abs(left_eye_height - right_eye_height)
            eyebrow_asymmetry = abs(left_eyebrow_height - right_eyebrow_height)

            features = [
                avg_eye_openness,  # Apertura de ojos
                mouth_height,  # Altura de boca
                mouth_width,  # Ancho de boca
                avg_eyebrow_height,  # Altura de cejas
                mouth_eye_ratio,  # Ratio boca/ojos
                mouth_aspect_ratio,  # Forma de boca
                eye_asymmetry,  # Asimetría ocular
                eyebrow_asymmetry  # Asimetría de cejas
            ]

            # Normalización conservadora
            features = np.clip(features, 0.001, 10.0)

        except Exception as e:
            print(f"❌ Error extrayendo características: {e}")
            # Valores neutrales por defecto
            features = [0.035, 0.025, 0.16, 0.08, 0.7, 6.5, 0.002, 0.003]

        return np.array(features)

    def add_training_sample(self, features, emotion):
        """Agregar muestra de entrenamiento"""
        if len(features) != 8:
            return False

        self.training_data["X"].append(features)
        self.training_data["y"].append(emotion)
        self.save_training_data()

        total = len(self.training_data["y"])
        count = self.training_data["y"].count(emotion)
        print(f"✅ {emotion} agregada. Total: {total}")

        return True

    def load_or_train_model(self):
        """Cargar o entrenar modelo"""
        if os.path.exists(self.model_file) and len(self.training_data["X"]) > 10:
            try:
                self.model = joblib.load(self.model_file)
                print("✅ Modelo cargado exitosamente")
                self.evaluate_model()
                return
            except Exception as e:
                print(f"❌ Error cargando modelo: {e}")

        # Entrenar si hay suficientes datos
        if len(self.training_data["X"]) >= 20:
            print("🔧 Entrenando modelo...")
            self.train_model()
        else:
            print("⚠️  Pocas muestras. Modelo no entrenado.")

    def train_model(self):
        """Entrenar modelo simple y robusto"""
        X = np.array(self.training_data["X"])
        y = np.array(self.training_data["y"])

        print(f"📚 Entrenando con {len(X)} muestras...")

        # Balancear clases
        unique_emotions, counts = np.unique(y, return_counts=True)
        print("📈 Distribución:")
        for emotion, count in zip(unique_emotions, counts):
            print(f"  {emotion}: {count}")

        if len(unique_emotions) > 1:
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=unique_emotions, y=y
            )
            weight_dict = dict(zip(unique_emotions, class_weights))
        else:
            weight_dict = 'balanced'

        # Modelo simple pero efectivo
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=weight_dict
        )

        # Entrenar con validación
        if len(X) >= 30:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            print(f"🎯 Precisión en prueba: {test_accuracy:.3f}")
        else:
            self.model.fit(X, y)

        joblib.dump(self.model, self.model_file)
        print("✅ Modelo entrenado y guardado")
        self.evaluate_model()

    def evaluate_model(self):
        """Evaluar modelo actual"""
        if self.model is None or len(self.training_data["X"]) == 0:
            return

        X = np.array(self.training_data["X"])
        y = np.array(self.training_data["y"])

        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)

        print(f"🏆 Precisión general: {accuracy:.3f}")

        emotions = sorted(set(y))
        print("📊 Por emoción:")
        for emotion in emotions:
            mask = y == emotion
            if np.sum(mask) > 0:
                emotion_acc = np.mean(y_pred[mask] == emotion)
                print(f"  {emotion}: {emotion_acc:.3f}")

    def predict_emotion(self, landmarks):
        """Predecir emoción con confianza"""
        if self.model is None:
            return "Neutral", 0.0

        try:
            features = self.extract_robust_features(landmarks)
            features = features.reshape(1, -1)

            probabilities = self.model.predict_proba(features)[0]
            emotion_labels = self.model.classes_

            best_idx = np.argmax(probabilities)
            best_emotion = emotion_labels[best_idx]
            best_confidence = probabilities[best_idx]

            # Solo retornar si la confianza es buena
            if best_confidence > 0.6:
                return best_emotion, best_confidence
            else:
                return "Neutral", best_confidence

        except Exception as e:
            return "Neutral", 0.0


def main():
    # Inicializar reconocedor
    recognizer = SimpleEmotionRecognizer()

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🎥 Iniciando reconocimiento de emociones...")
    print("💡 Presiona '1-5' para entrenar: 1=Feliz, 2=Triste, 3=Enojado, 4=Sorprendido, 5=Neutral")
    print("💡 Presiona 't' para reentrenar modelo")
    print("💡 Presiona 'q' para salir\n")

    current_emotion = "Neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = recognizer.face_mesh.process(rgb_frame)

        emotion = "Neutral"
        confidence = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            emotion, confidence = recognizer.predict_emotion(landmarks)

            # Dibujar landmarks (opcional)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # Mostrar resultado
        color = (0, 255, 0)  # Verde por defecto
        if emotion == "Feliz":
            color = (0, 255, 0)  # Verde
        elif emotion == "Triste":
            color = (255, 0, 0)  # Azul
        elif emotion == "Enojado":
            color = (0, 0, 255)  # Rojo
        elif emotion == "Sorprendido":
            color = (0, 255, 255)  # Amarillo

        cv2.putText(frame, f"Emocion: {emotion} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "1=Feliz 2=Triste 3=Enojado 4=Sorprendido 5=Neutral",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "t=Reentrenar q=Salir",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Reconocimiento de Emociones', frame)

        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('t') and recognizer.model is not None:
            print("🔄 Reentrenando modelo...")
            recognizer.train_model()
        elif results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            if key == ord('1'):  # Feliz
                features = recognizer.extract_robust_features(landmarks)
                recognizer.add_training_sample(features, "Feliz")
            elif key == ord('2'):  # Triste
                features = recognizer.extract_robust_features(landmarks)
                recognizer.add_training_sample(features, "Triste")
            elif key == ord('3'):  # Enojado
                features = recognizer.extract_robust_features(landmarks)
                recognizer.add_training_sample(features, "Enojado")
            elif key == ord('4'):  # Sorprendido
                features = recognizer.extract_robust_features(landmarks)
                recognizer.add_training_sample(features, "Sorprendido")
            elif key == ord('5'):  # Neutral
                features = recognizer.extract_robust_features(landmarks)
                recognizer.add_training_sample(features, "Neutral")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()