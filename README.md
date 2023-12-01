# desafiolapiscoia
resolução do desafio para bolsista do Lapisco

Candidato: Tales Rodrigues

Descrição do Desafio:

Implementar um scrpit em python para detectar rostos em um vídeo utilizando YOLOv8.
Além disso, é necessário incorporar a funcionalidade de applicar um efeito de borramento nos rostos detectados, além de contar o número de pessoas no vídeo.

Para realização da task foi utilizado um modelo já treinado disponibilizado publicamente "yolov8n-face.pt"

-- Instruções de Execução do Script:
1 - Clonar esse repositório

    git clone https://github.com/tales138/desafiolapiscoia.git

2 - Instalar a biblioteca ultralytics que permite usar YOLOv8

    pip install ultralytics
    
3 - Caso seu sistema não tenha a biblioteca opencv

    pip install opencv-python

4 - Caso deseje excutar o script em outro arquivo de vídeo, é necessário atualizar a variável "video_path" com o caminho do arquivo.
