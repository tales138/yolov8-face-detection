from ultralytics import YOLO
import cv2

# carregando o modelo pré-treinado yolov8n-face.pt 
model = YOLO("yolov8l-face.pt")

# carregando o vídeo de exemplo
video_path = 'demo1.mp4'

#variável que armazena o vídeo capturado pela opencv
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Erro ao abrir o vídeo. Certifique-se de que o caminho do vídeo está correto.")
else:    
    controle = True
    # read frames
    while capture.isOpened():
        controle, frame = capture.read()
        
        contador = 0 #contador para o número de pessoas no vídeo
        if controle:
            results = model.track(frame, persist=True) #instanciamento do modelo
            boxes = results[0].boxes.cpu().numpy() #retorna os objetos que representam as faces no vídeo
            xyxy = boxes.xyxy #obtendo o conjunto de coordenadas das faces das pessoas no vídeo

            for coordenada in xyxy:
                contador += 1
                x1, y1, x2, y2 = coordenada #obtenção de cada coordenada
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1) #borramento de face

            frame = results[0].plot() #plotagem do frame de enquadramento da face

            cv2.rectangle(frame, (0, 0), (270, 50), (0, 0, 0), -1)#frame com o número de pessoas no vídeo
            cv2.putText(frame, f'Qtd Pessoas: {contador}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)

            # visualize
            cv2.imshow('frame',frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
