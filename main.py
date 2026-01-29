import shadow as sh
import modelos_ml as mml
import divisao_dados as dd
from modelos_ml_he import ModelosML_HE as ml_he


###########################################
#           División de los datos         #
###########################################

divisao_dados = dd.DivisaoDados()

divisao_dados.carregar_dados()

x_train_full, y_train_full = divisao_dados.tfds_to_numpy(divisao_dados.treino)
x_test_full, y_test_full = divisao_dados.tfds_to_numpy(divisao_dados.teste)

x_all = divisao_dados.concatenar(x_train_full, x_test_full)
y_all = divisao_dados.concatenar(y_train_full, y_test_full)

x_all, y_all = divisao_dados.shuffle_data(x_all, y_all)

x_target, x_shadow = divisao_dados.split_data_target_shadow(x_all)
y_target, y_shadow = divisao_dados.split_data_target_shadow(y_all)

x_target_treino, x_target_teste = divisao_dados.split_data_target(x_target)
y_target_treino, y_target_teste = divisao_dados.split_data_target(y_target)

##############################################
#     Trabajando con Random Forest com HE    #
##############################################

rf_he = ml_he.RF_HE(x_target_treino, y_target_treino)
'''
###########################################
#       Trabajando con Random Forest      #
###########################################

modelo_rf = mml.ModelosML.RF(x_target_treino, x_target_teste)

modelo_rf.configuracao_modelo()

modelo_rf.treinamento_modelo(y_target_treino)

modelo_rf.pontos_acuracia(y_target_treino, y_target_teste)

###########################################
#           Trabajando con MLP            #
###########################################

modelo_mlp = mml.ModelosML.MLP(x_target_treino, y_target_treino)

modelo_mlp.configuracao_modelo()

modelo_mlp.treinamento_modelo(x_target_teste, y_target_teste)

modelo_mlp.pontos_acuracia(x_target_teste, y_target_teste)

###########################################
#           Trabajando con CNN            #
###########################################

modelo_cnn = mml.ModelosML.CNN(x_target_treino, y_target_treinox_target_treino, y_target_treino)

modelo_cnn.configuracao_modelo()

modelo_cnn.treinamento_modelo(x_target_teste, y_target_teste)

modelo_cnn.pontos_acuracia(x_target_teste, y_target_teste)
'''
###########################################
#              Shadow Attacks             #
###########################################

data_shadow = sh.ShadowAttack(x_shadow, y_shadow)
data_shadow.create_shadow_datasets()

#ataque_shadow_rf = sh.Shadow_RF()
#ataque_shadow_rf.treinamento_modelo(data_shadow.datasets)

ataque_shadow_mlp = sh.Shadow_MLP()
ataque_shadow_mlp.treinamento_modelo(data_shadow.datasets)

#ataque_shadow_cnn = sh.Shadow_CNN()
#ataque_shadow_cnn.treinamento_modelo(data_shadow.datasets)
