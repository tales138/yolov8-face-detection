from ultralytics import YOLO
import cv2
import numpy as np

# Função para calcular a distância
def calcular_distancia(tamanho_real_face, distancia_focal, tamanho_face_imagem):
    if tamanho_face_imagem == 0:
        return None
    return (tamanho_real_face * distancia_focal) / tamanho_face_imagem

# Função para calcular as coordenadas geográficas estimadas
def estimar_coordenadas(lat_camera, lon_camera, distancia, angulo_azimute=0):
    # Converter distância de metros para graus de latitude e longitude
    delta_lat = distancia / 111000
    delta_lon = distancia / (111000 * np.cos(np.radians(lat_camera)))
    
    # Ajustar longitude com base no ângulo de azimute (simplificação)
    lon_nova = lon_camera + delta_lon * np.cos(np.radians(angulo_azimute))
    lat_nova = lat_camera + delta_lat * np.sin(np.radians(angulo_azimute))
    
    return lat_nova, lon_nova

# Carregar o modelo pré-treinado
model = YOLO("yolov8l-face.pt")

# Definir a altura real da face (em cm), a distância focal (em pixels) e a posição da câmera
altura_real_face = 20  # Altura real da face em cm
distancia_focal = 800  # Distância focal da câmera (exemplo; você deve calibrar a sua câmera)
lat_camera = -23.5505  # Latitude da câmera
lon_camera = -46.6333  # Longitude da câmera

# Carregar o vídeo de exemplo
video_path = 'demo1.mp4'

# Captura de vídeo com OpenCV
capture = cv2.VideoCapture(0)  # Use 0 para câmera ao vivo, ou caminho para um vídeo
if not capture.isOpened():
    print("Erro ao abrir o vídeo. Certifique-se de que o caminho do vídeo está correto.")
else:    
    while capture.isOpened():
        controle, frame = capture.read()
        if controle:
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.cpu().numpy()
            xyxy = boxes.xyxy

            contador = 0  # Contador para o número de pessoas no vídeo

            for coordenada in xyxy:
                x1, y1, x2, y2 = coordenada
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Calcular a altura da face na imagem
                altura_face_imagem = y2 - y1
                
                # Calcular a distância
                distancia = calcular_distancia(altura_real_face, distancia_focal, altura_face_imagem)
                
                # Estimar as coordenadas da pessoa
                lat_pessoa, lon_pessoa = estimar_coordenadas(lat_camera, lon_camera, distancia)
                
                # Contar número de faces detectadas
                contador += 1

                # Desenhar a caixa delimitadora
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                
                # Criar o texto com distância e coordenadas
                texto_info = f'{distancia:.2f} cm, Lat: {lat_pessoa:.4f}, Lon: {lon_pessoa:.4f}'
                
                # Calcular a posição do texto
                (w, h), _ = cv2.getTextSize(texto_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                x_text = x1 + (x2 - x1 - w) // 2
                y_text = y1 + (y2 - y1 + h) // 2
                
                # Desenhar o texto no centro da caixa delimitadora
                cv2.putText(frame, texto_info, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 2)

            # Plotar o frame com a contagem de pessoas
            frame = results[0].plot()
            cv2.rectangle(frame, (0, 0), (270, 50), (0, 0, 0), -1)
            cv2.putText(frame, f'Qtd Pessoas: {contador}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)

            # Visualizar
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
