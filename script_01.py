# ==============================================================
# Contagem e classificação de veículos 
# ==============================================================
# Importando as bibliotecas

import cv2
import csv
import collections
import numpy as np
from tracker import *
from time import sleep
from numba import jit
import cProfile

# ==============================================================
# Importando o vídeo
delay = 60  # FPS do vídeo

# Iniciar o rastreador
tracker = EuclideanDistTracker()

# Iniciar a captura de vídeo
cap = cv2.VideoCapture('video.mp4')
input_size = 320
 
# Detectando a entrada de veículos
conf_entrada = 0.2
nms_entrada= 0.2

# Cor respectiva as linhas de detecção
font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Posição das linhas de detecção intermediárias
linha_mediana = 225   
linha_superior = linha_mediana - 15
linha_inferior = linha_mediana + 15

# ==============================================================
# Pacotes contendo a programação orientada a objetos

# Armazenamento dos nomes da lista "Coco"
classe_arq = "coco.names"
classe_nomes = open(classe_arq).read().strip().split('\n')
#print(classe_nomes)
#print(len(classe_nomes))

# Lista de classes para detecção das mesmas
requeridas_classes = [2, 3, 5, 7]     # detectar apenas motos, carros, caminhoes, onibus 

classes_detectadas = []

# Arquivos de modelos
modelo_config = 'yolov3-320.cfg'  # You Only Look Once detectar e identificar classes
modelo_formato = 'yolov3-320.weights'

# Configuração dos modelos de trabalho
net = cv2.dnn.readNetFromDarknet(modelo_config, modelo_formato)

# Configuração de trabalho dentro do processsamento melhorar performance de processamento

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # para as unidades graficas de processamento em GPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Definir cores aleatorias para cada classe
np.random.seed(42)
cores = np.random.randint(0, 255, size=(len(classe_nomes), 3), dtype='uint8')

# ==============================================================
# Encontrado o centro do retangulo
def local_centro(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

# ==============================================================    
# Armazenar as informações de contagem de veículos
tracker_lista_dest = []
tracker_lista_orig = []
lista_dest = [0, 0, 0, 0]
lista_orig = [0, 0, 0, 0]

# ==============================================================
# Contador de veículos
def conta_veic(box_id, img):

    x, y, w, h, id, index = box_id

    # Encontrar o centro de deteccao
    centro = local_centro(x, y, w, h)
    ix, iy = centro
    
    # Encontrar a atual posicao do veiculo
    if (iy > linha_superior) and (iy < linha_mediana):

        if id not in tracker_lista_dest:
            tracker_lista_dest.append(id)

    elif iy < linha_inferior and iy > linha_mediana:
        if id not in tracker_lista_orig:
            tracker_lista_orig.append(id)
            
    elif iy < linha_superior:
        if id in tracker_lista_orig:
            tracker_lista_orig.remove(id)
            lista_dest[index] = lista_dest[index]+1

    elif iy > linha_inferior:
        if id in tracker_lista_dest:
            tracker_lista_dest.remove(id)
            lista_orig[index] = lista_orig[index] + 1

    # Formulando  o retangulo com base no contorno do veiculo 
    cv2.circle(img, centro, 2, (0, 0, 255), -1)  # end here
    # print(lista_dest, lista_orig)

# ==============================================================
# Encontar objetos detectados dentro dos arquivos
def postProcess(outputs,img):
    global classes_detectadas 
    height, width = img.shape[:2]
    boxes = []
    classes_ident = []
    confidence_scores = []
    detec = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in requeridas_classes:
                if confidence > conf_entrada:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classes_ident.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression (a ideia é identificar um objeto por vez)
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, conf_entrada, nms_entrada)
    # print(classes_ident)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in cores[classes_ident[i]]]
            name = classe_nomes[classes_ident[i]]
            classes_detectadas.append(name)
            # Desenhar o nome das classes
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Desenhar retangulo de contorno
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detec.append([x, y, w, h, requeridas_classes.index(classes_ident[i])])

        # Atualizar o tracker de cada objeto
        boxes_ids = tracker.update(detec)
        for box_id in boxes_ids:
            conta_veic(box_id, img)
# ==============================================================
# Avaliacao em tempo real
boxes = []
classes_ident = []
confidence_scores = []
detec = []
# Aplicar os boxes
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, conf_entrada, nms_entrada)
# print(classes_ident)
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        print(x,y,w,h)
        
# ==============================================================
# Avaliacao em tempo real
def realTime():
    while True:
        #tempo = float(1/delay)
        #sleep(tempo)  # Dá um delay entre cada processamento 
        success, img = cap.read() # Le o frame do video
        
        i=0 #Contador de frames ###
        frameTime = 1 # tempo de cada frame
        
        if success:  # Se existirem frames, continue
        
            img = cv2.resize(img,(0,0),None,0.5,0.5)
            ih, iw, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Estabelecer inputs 
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            
            # Saida dos dados
            outputs = net.forward(outputNames)
        
            # Encontrar e identificar os objetos
            postProcess(outputs,img)

            # Desenhar as linhas de cruzamentos

            cv2.line(img, (0, linha_mediana), (iw, linha_mediana), (255, 0, 255), 2)
            cv2.line(img, (0, linha_superior), (iw, linha_superior), (0, 0, 255), 2)
            cv2.line(img, (0, linha_inferior), (iw, linha_inferior), (0, 0, 255), 2)

            # Mostrando os frames
            cv2.imshow('Output', img)

            ret = cap.grab() #pega o frame
            i=i+1 #incrementa o contador 
            if i % 3 == 0: # mostra um terco dos frames (acelerar)
                ret, frame = cap.retrieve() #decodifica os frames
                cv2.imshow('frame',img)
                if cv2.waitKey(frameTime) & 0xFF == ord('q'):
                    break
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        else:  # acabaram os frames

            break               
                                  
    # ==============================================================
    # Salvar as informacoes 

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direcao', 'Carro', 'Moto', 'Onibus', 'Caminhao'])
        lista_dest.insert(0, "Destino")
        lista_orig.insert(0, "Origem")
        cwriter.writerow(lista_dest)
        cwriter.writerow(lista_orig)
    f1.close()
    # print("Arquivo salvo 'data.csv'")
    # Termina de ler o arquivo de video e fecha a janela
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':  # controla a execucao do arquivo importado (sem rodar)
    realTime()
    #from_static_image(image_file)
    
    
