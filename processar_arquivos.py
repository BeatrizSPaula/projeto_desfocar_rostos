import os
import cv2
from ultralytics import YOLO
model = YOLO('face_yolov8n_v2.pt')

def blur_faces_in_images(image_paths: list, output_dir: str, blur_kernel: tuple = (51, 51), confidence_threshold: float = 0.5):
    """
    Detecta rostos em uma lista de imagens e os borra, salvando as imagens processadas
    em um diretório de saída.

    Args:
        image_paths (list): Uma lista de caminhos para os arquivos de imagem de entrada.
        output_dir (str): O caminho para o diretório onde as imagens de saída serão salvas.
        blur_kernel (tuple): O tamanho do kernel para o desfoque gaussiano.
                             Valores maiores resultam em mais desfoque. Deve ser uma tupla de ímpares.
        confidence_threshold (float): Limite de confiança para a detecção de rostos.
                                      Apenas rostos detectados com confiança acima deste valor serão borrados.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório de saída criado em: {output_dir}")

    try:
        model = YOLO('face_yolov8n_v2.pt')
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO. Certifique-se de que 'face_yolov8n_v2.pt' está disponível. Erro: {e}")
        return
        
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Aviso: O arquivo de imagem não foi encontrado: {image_path}. Pulando...")
            continue
            
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"blurred_{image_name}")
        print(f"Processando {image_name}...")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Erro: Não foi possível ler a imagem {image_path}")
            continue
            
        height, width, _ = frame.shape
            
        results = model(frame, verbose=False)
            
        for r in results:
            for box in r.boxes:
                confidence = box.conf[0]
                
                if confidence > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        blurred_face_roi = cv2.GaussianBlur(face_roi, blur_kernel, 0)
                        
                        frame[y1:y2, x1:x2] = blurred_face_roi
                        
        cv2.imwrite(output_path, frame)
        print(f"Processamento concluído para {image_name}. Salvo em: {output_path}")
        
    print("\nTodas as imagens foram processadas com sucesso!")


def blur_faces_in_videos(video_paths: list, output_dir: str, blur_kernel: tuple = (51, 51), confidence_threshold: float = 0.5, detection_interval: int = 5):
    """
    Detecta rostos em uma lista de vídeos e os borra, salvando os vídeos processados
    em um diretório de saída.

    Args:
        video_paths (list): Uma lista de caminhos para os arquivos de vídeo de entrada (.mp4).
        output_dir (str): O caminho para o diretório onde os vídeos de saída serão salvos.
        blur_kernel (tuple): O tamanho do kernel para o desfoque gaussiano.
                             Valores maiores resultam em mais desfoque. Deve ser uma tupla de ímpares (ex: (51, 51)).
        confidence_threshold (float): Limite de confiança para a detecção de rostos.
                                      Apenas rostos detectados com confiança acima deste valor serão borrados.
        detection_interval (int): Intervalo de quadros para realizar a detecção de rostos.
                                  Por exemplo, 5 significa detectar a cada 5 quadros.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório de saída criado em: {output_dir}")

    try:
        model = YOLO('face_yolov8n_v2.pt')
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO. Certifique-se de que 'face_yolov8n_v2.pt' está disponível. Erro: {e}")
        return

    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Aviso: O arquivo de vídeo não foi encontrado: {video_path}. Pulando...")
            continue

        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"blurred_{video_name}")
        print(f"Processando {video_name}...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_number = 0
        last_known_faces = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % 30 == 0:
                print(f"  -> Processando frame {frame_number}/{frame_count} de {video_name}")

            if frame_number % detection_interval == 0:
                results = model(frame, verbose=False)
                last_known_faces = []
                for r in results:
                    for box in r.boxes:
                        if box.conf[0] > confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            last_known_faces.append((x1, y1, x2, y2))
            
            for (x1, y1, x2, y2) in last_known_faces:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size > 0:
                    blurred_face_roi = cv2.GaussianBlur(face_roi, blur_kernel, 0)
                    frame[y1:y2, x1:x2] = blurred_face_roi

            out.write(frame)

        cap.release()
        out.release()
        print(f"Processamento concluído para {video_name}. Salvo em: {output_path}")

    print("\nTodos os vídeos foram processados com sucesso!")

if __name__ == '__main__':
    print("--- Exemplo de Demonstração ---")
    print("Execute 'pip install opencv-python ultralytics' se ainda não o fez.")
    print("Crie um diretório de teste e adicione alguns arquivos para testar as funções.")
    print("Em seguida, descomente as chamadas das funções no bloco '__main__'.")

    # Diretórios de entrada e saída
    input_videos_dir = 'input_videos'
    input_images_dir = 'input_images'
    output_videos_dir = 'blurred_videos'
    output_images_dir = 'blurred_images'

    # Cria diretórios de entrada se não existirem
    if not os.path.exists(input_videos_dir):
        os.makedirs(input_videos_dir)
        print(f"Diretório criado: {input_videos_dir} (adicione vídeos .mp4 para testar)")
    if not os.path.exists(input_images_dir):
        os.makedirs(input_images_dir)
        print(f"Diretório criado: {input_images_dir} (adicione imagens .jpg/.png para testar)")

    # Lista arquivos válidos
    input_videos = [os.path.join(input_videos_dir, f) for f in os.listdir(input_videos_dir)
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    input_images = [os.path.join(input_images_dir, f) for f in os.listdir(input_images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Só roda se houver arquivos
    if input_videos:
        blur_faces_in_videos(input_videos, output_videos_dir, blur_kernel=(75, 75), confidence_threshold=0.6, detection_interval=10)
    else:
        print("Nenhum vídeo encontrado em input_videos.")

    if input_images:
        blur_faces_in_images(input_images, output_images_dir, blur_kernel=(75, 75), confidence_threshold=0.6)
    else:
        print("Nenhuma imagem encontrada em input_images.")