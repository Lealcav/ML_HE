import tensorflow.keras as keras
from keras import layers, models
from keras.src.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Pyfhel import Pyfhel

class ModelosML:
    class RF:
        def __init__(self, x_treino, x_teste):
            self.treino_flat, self.teste_flat = x_treino.reshape(len(x_treino), -1), x_teste.reshape(len(x_teste), -1)
        
        def configuracao_modelo(self):
            self.modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)

        def treinamento_modelo(self, y_target_treino):
            self.modelo.fit(self.treino_flat, y_target_treino)

        def pontos_acuracia(self, y_target_train, y_target_test):
            acc_treino = accuracy_score(y_target_train, self.modelo.predict(self.treino_flat))
            acc_teste = accuracy_score(y_target_test, self.modelo.predict(self.teste_flat))
            print(f"RF Train Accuracy: {acc_treino*100:.2f}%")# {acc_treino*100:.2f}%")
            print(f"RF Test Accuracy: {acc_teste*100:.2f}%")#{acc_teste*100:.2f}%")

    class MLP:
        def __init__(self, x_treino, y_treino):
            self.treino_x, self.treino_y = x_treino, y_treino

        def configuracao_modelo(self):
            self.modelo = model = models.Sequential([
                layers.Flatten(input_shape=(28, 28, 1)),

                # Camada de entrada com mais neurônios
                layers.Dense(1024, activation='relu', kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),

                # Camadas ocultas progressivamente menores
                layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),

                layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),

                # Camada de saída
                layers.Dense(10, activation='softmax')
            ])

            # Optimizer com learning rate ajustado
            optimizer = keras.optimizers.Adam(learning_rate=0.001)

            self.modelo.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        def treinamento_modelo(self, x_target_test=None, y_target_test=None):
            self.modelo.fit(
                self.treino_x, self.treino_y,
                epochs=15,
                batch_size=128,
                validation_data=(x_target_test, y_target_test) if x_target_test is not None else None,
                verbose=1
            )

        def pontos_acuracia(self, x_test, y_test):
            # Avalia usando as ferramentas do Keras
            loss_tr, acc_treino = self.modelo.evaluate(self.treino_x, self.treino_y, verbose=0)
            loss_te, acc_teste = self.modelo.evaluate(x_test, y_test, verbose=0)
            print(f"MLP Train Accuracy: {acc_treino*100:.2f}%")
            print(f"MLP Test Accuracy: {acc_teste*100:.2f}%")

    class CNN:
        def __init__(self, x_treino, y_treino):
            self.treino_x, self.treino_y = x_treino, y_treino
        
        def configuracao_modelo(self):
            self.modelo = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])

            self.modelo.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        def treinamento_modelo(self, x_target_test=None, y_target_test=None):
            self.modelo.fit(
                self.treino_x, self.treino_y,
                epochs=15,
                batch_size=128,
                validation_data=(x_target_test, y_target_test) if x_target_test is not None else None,
                verbose=1
            )

        def pontos_acuracia(self, x_test, y_test):
            # Avalia usando as ferramentas do Keras
            loss_tr, acc_treino = self.modelo.evaluate(self.treino_x, self.treino_y, verbose=0)
            loss_te, acc_teste = self.modelo.evaluate(x_test, y_test, verbose=0)
            print(f"MLP Train Accuracy: {acc_treino*100:.2f}%")
            print(f"MLP Test Accuracy: {acc_teste*100:.2f}%")
