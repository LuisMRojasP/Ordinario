"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA:Redes neuronales
Tema: Proyecto ordinario
Alumnos: Rojas Palacios Luis Martin
Profesor:Lopez Chau Asdrubal
Descripción: Identificador de sonidos

Created on Tue Dec 3 18:38:04 2021

@author: Luis Martin R.P
"""
import numpy as np
import os
import wave
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def datosentre(matriz,xi,xn):
    xin=matriz[:,xi:xn+1]
    return xin

path1 = './WAVmoto'
path2 = "./WAVcarro"
pathsmoto = os.listdir(path1)
pathscarro = os.listdir(path2)
moto_paths = []
carro_paths = []
datas = pd.DataFrame(columns=['canales', 'tamaño de muestra','tasa de frames','numero de frames','Etiqueta'])

for i in pathsmoto:
    moto_paths.append(path1+"/"+i)
#print(moto_paths)

for i in pathscarro:
    carro_paths.append(path2+"/"+i)
#print(carro_paths)

for i in moto_paths:
    wav_file = wave.open(i,'r')
    numcanales = wav_file.getnchannels()# numero de canales
    tammuestra = wav_file.getsampwidth()# tamaño de muestra
    tasafra = wav_file.getframerate()# tasa de frames
    numfra = wav_file.getnframes()# numero de frames
    datas.loc[i]=[numcanales,tammuestra,tasafra,numfra,0]

for i in carro_paths:
    wav_file = wave.open(i,'r')
    numcanales = wav_file.getnchannels()# numero de canales
    tammuestra = wav_file.getsampwidth()# tamaño de muestra
    tasafra = wav_file.getframerate()# tasa de frames
    numfra = wav_file.getnframes()# numero de frames
    datas.loc[i]=[numcanales,tammuestra,tasafra,numfra,1]
    
datas.to_csv('datos.csv')
datos=pd.read_csv("datos.csv")
motos=datos[datos["Etiqueta"]==0]
carros=datos[datos["Etiqueta"]==1]
#print("Motos",motos,"Carros",carros)
plt.scatter(motos["tasa de frames"],motos["numero de frames"],marker=".",s=50,color="skyblue",label="Moto")
plt.scatter(carros["tasa de frames"],carros["numero de frames"],marker="*",s=50,color="red",label="Carro")
plt.ylabel("Numero de frames")
plt.xlabel("Tasa de frames")
plt.legend(bbox_to_anchor=(1,.2))
plt.show()

prep=datos[["tasa de frames","numero de frames"]]
clase=datos["Etiqueta"]
escalador=preprocessing.MinMaxScaler()
prep=escalador.fit_transform(prep)
#print(prep)

clasificador=KNeighborsClassifier(n_neighbors=5)
clasificador.fit(prep,clase)

pvalid=pd.read_csv("datospp.csv")
matrizdatos=np.array(pvalid)
x_ini=0
x_fin=4
xe=(datosentre(matrizdatos,x_ini,x_fin))
d=len(xe)
#print(xe)
predicciones = pd.DataFrame(columns=['Prediccion'])
for i in range(d):
    tafra=xe[i,3]
    nufra=xe[i,4]
    #print("tasa de frames: ",tafra,"\tnumero de frames: ",nufra)
    audio=escalador.transform([[tafra,nufra]])
    #print("Clase",clasificador.predict(audio))
    #print("Probabilidades por clase: ",clasificador.predict_proba(audio))
    if clasificador.predict(audio) == 0:
        predicciones.loc[xe[i,0]]=["Moto"]
    else:
        predicciones.loc[xe[i,0]]=["Carro"]
predicciones.to_csv("predicciones.csv")
