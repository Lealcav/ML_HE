import modelos_ml as mml
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import numpy as np

class ShadowAttack:
    def __init__(self, x_shadow_in, y_shadow_in, n_models=3):
        self.x_shadow, self.y_shadow = x_shadow_in, y_shadow_in
        self.n_models = n_models
        self.datasets = None

    def create_shadow_datasets(self):
        datasets = []
        samples_per_model = len(self.x_shadow) // self.n_models

        for i in range(self.n_models):
            start_idx = i * samples_per_model
            end_idx = start_idx + samples_per_model

            x_shadow_model = self.x_shadow[start_idx:end_idx]
            y_shadow_model = self.y_shadow[start_idx:end_idx]

            # 75/25 split
            split = int(0.75 * len(x_shadow_model))
            x_train = x_shadow_model[:split]
            y_train = y_shadow_model[:split]
            x_test = x_shadow_model[split:]
            y_test = y_shadow_model[split:]

            datasets.append((x_train, y_train, x_test, y_test))

        self.datasets = datasets

    def preparar_dados_ataque(shadow_models, shadow_datasets):
        x_ataque, y_ataque = [], []
    
        for i, modelo in enumerate(shadow_models):
            x_tr, y_tr, x_te, y_te = shadow_datasets[i]
        
            # 1. Predições de membros (estavam no treino do shadow)
            # Para RF use predict_proba; para Keras use predict
            prob_membros = modelo.predict_proba(x_tr.reshape(len(x_tr), -1)) 
            
            x_ataque.append(prob_membros)
            y_ataque.append(np.ones(len(prob_membros))) # 1 = Membro

            # 2. Predições de não-membros (não estavam no treino)
            prob_nao_membros = modelo.predict_proba(x_te.reshape(len(x_te), -1))
            x_ataque.append(prob_nao_membros)
            y_ataque.append(np.zeros(len(prob_nao_membros))) # 0 = Não Membro

        return np.vstack(x_ataque), np.concatenate(y_ataque)
    
class Shadow_RF:
    def __init__(self):
        self.shadow_rf_models = []

    def treinamento_modelo(self, shadow_datasets):
        for i, (x_tr, y_tr, x_te, y_te) in enumerate(shadow_datasets):
            # 1. Preparação dos dados (Flatten para Random Forest)
            x_tr_flat = x_tr.reshape(len(x_tr), -1)
            x_te_flat = x_te.reshape(len(x_te), -1)

            # 2. Criação e Treino do modelo
            shadow_rf = RandomForestClassifier(n_estimators=100, random_state=42+i, n_jobs=-1, max_depth=20)
            shadow_rf.fit(x_tr_flat, y_tr)

            # 3. Cálculo da Acurácia usando os dados DESTA partição (i)
            # Usamos predict() nos dados de treino e teste desta fatia específica
            acc_treino = accuracy_score(y_tr, shadow_rf.predict(x_tr_flat))
            acc_teste = accuracy_score(y_te, shadow_rf.predict(x_te_flat))

            print(f"--- Shadow RF Modelo {i+1} ---")
            print(f"Acurácia Treino: {acc_treino*100:.2f}%")
            print(f"Acurácia Teste (Validação): {acc_teste*100:.2f}%")

            # 4. Salva o modelo treinado na lista da classe
            self.shadow_rf_models.append(shadow_rf)

    def preparar_dados_ataque(shadow_models, shadow_datasets):
        x_ataque, y_ataque = [], []
    
        for i, modelo in enumerate(shadow_models):
            x_tr, y_tr, x_te, y_te = shadow_datasets[i]
        
            # 1. Predições de membros (estavam no treino do shadow)
            # Para RF use predict_proba; para Keras use predict
            prob_membros = modelo.predict_proba(x_tr.reshape(len(x_tr), -1)) 
            
            x_ataque.append(prob_membros)
            y_ataque.append(np.ones(len(prob_membros))) # 1 = Membro

            # 2. Predições de não-membros (não estavam no treino)
            prob_nao_membros = modelo.predict_proba(x_te.reshape(len(x_te), -1))
            x_ataque.append(prob_nao_membros)
            y_ataque.append(np.zeros(len(prob_nao_membros))) # 0 = Não Membro

        return np.vstack(x_ataque), np.concatenate(y_ataque)

class Shadow_MLP:
    def __init__(self):
        self.shadow_mlp_models = []
            
    def treinamento_modelo(self, shadow_datasets):
        for i, (x_tr, y_tr, x_te, y_te) in enumerate(shadow_datasets):
            shadow_mlp = mml.ModelosML.MLP(x_tr, y_tr)
            shadow_mlp.configuracao_modelo()
            shadow_mlp.treinamento_modelo(x_te, y_te)

            print("Shadown MLP")
            shadow_mlp.pontos_acuracia(x_te, y_te)

            self.shadow_mlp_models.append(shadow_mlp)

class Shadow_CNN:
    def __init__(self):
        self.shadow_cnn_models = []

    def treinamento_modelo(self, shadow_datasets):
        for i, (x_tr, y_tr, x_te, y_te) in enumerate(shadow_datasets):
            instancia_cnn = mml.ModelosML.CNN(x_tr, y_tr)
            instancia_cnn.configuracao_modelo()
            shadow_cnn = instancia_cnn.modelo
            
            shadow_cnn.fit(x_tr, y_tr, epochs=30, batch_size=128, verbose=0, validation_data=(x_te, y_te))