import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import chess
import chess.svg
import cairosvg
from shapely.geometry import Polygon
import os
import argparse
import sys
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from io import BytesIO
os.environ["XDG_SESSION_TYPE"] = "xcb"
photo = None


def order_points(pts):
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def plot_grid_on_transformed_image(image):
    corners = np.array([[0, 0],
                        [image.size[0], 0],
                        [0, image.size[1]],
                        [image.size[0], image.size[1]]])

    corners = order_points(corners)
    figure(figsize=(10, 10), dpi=80)

    # im = plt.imread(image)
    implot = plt.imshow(image)

    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - y0) / 8
        pts = [(x0 + i * dx, y0 + i * dy) for i in range(9)]
        return pts

    ptsT = interpolate(TL, TR)
    ptsL = interpolate(TL, BL)
    ptsR = interpolate(TR, BR)
    ptsB = interpolate(BL, BR)

    for a, b in zip(ptsL, ptsR):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
    for a, b in zip(ptsT, ptsB):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")

    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL


def draw_boxes(image, detections):
    # Suponemos que `image` es un array de numpy de la imagen
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        color = (255, 0, 0)  # Color rojo
        thickness = 2  # Espesor de la línea de la caja
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def chess_pieces_detector(image, cont, direc_path):
    save_dir = f"{direc_path}/seccion{cont}"
    os.makedirs(save_dir, exist_ok=True)
    model_trained = YOLO("/home/carlos/PycharmProjects/ReconocerEsquinas/modelos/Yolov8xpiezas.pt")
    results = model_trained.predict(
        source=image,
        line_width=1,
        conf=0.3,
        augment=True,
        save_txt=True,
        save=True,
        show=True,
        iou=0.3,
        save_dir=save_dir
    )
    for result in results:
        result_image_path = os.path.join(save_dir, f"predict_seccion_{cont}.jpg")
        result.save(filename=result_image_path)  # Save annotated image
    boxes = results[0].boxes
    detections = boxes.xyxy.numpy()  # Asegúrate de que esto también devuelva un array de numpy
    return detections, boxes


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def connect_square_to_detection(detections, square, boxes):
    di = {0: 'b', 1: 'B', 2: 'K',
          3: 'k', 4: 'n', 5: 'N',
          6: 'p', 7: 'P', 8: 'q',
          9: 'Q', 10: 'r', 11: 'R'}

    list_of_iou = []

    for i in detections:

        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]

        # cut high pieces
        if box_y4 - box_y1 > 40:
            box_complete = np.array([[box_x1, box_y1 + 40], [box_x2, box_y2 + 40], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1, box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])

        # until here

        list_of_iou.append(calculate_iou(box_complete, square))

    num = list_of_iou.index(max(list_of_iou))

    piece = boxes[num]
    if max(list_of_iou) > 0.25:
        piece = boxes[num]
        return di[piece]
    else:
        piece = "empty"
        return piece

def four_point_transform(image, pts):
    img = Image.open(image)
    image = np.array(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    img = Image.fromarray(warped, "RGB")
    # img.show()
    # return the warped image
    return img, M


def detect_corners(image):
    # YOLO model trained to detect corners on a chessboard
    model_trained = YOLO("/home/carlos/PycharmProjects/ReconocerEsquinas/modelos/Yolov8Esquinas.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.01, save_txt=True, save=True)

    # get the corners coordinates from the model
    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:, 0:2]

    corners = order_points(points)

    return corners


def update_fen(fen, board):
    try:
        board.set_fen(fen)
        update_board(board)
    except ValueError as e:
        print(f"Error setting FEN: {e}")


# Definición de la función update_board

def update_board(board, label):
    try:
        svg_data = chess.svg.board(board)
        print("SVG data generated successfully")
    except Exception as e:
        print(f"Failed to generate SVG data: {e}")
        return

    try:
        png_image = cairosvg.svg2png(bytestring=svg_data)
        print("PNG image generated successfully")
    except Exception as e:
        print(f"Failed to convert SVG to PNG: {e}")
        return

    try:
        image = Image.open(BytesIO(png_image))
        image.save("temp.png")  # Guardar temporalmente la imagen para leerla en QPixmap
        print("Image saved successfully")
    except Exception as e:
        print(f"Failed to load and save image: {e}")
        return

    try:
        pixmap = QPixmap("temp.png")
        label.setPixmap(pixmap)
        print("Pixmap created and set successfully")
    except Exception as e:
        print(f"Failed to create Pixmap: {e}")
        return

    print(f"PhotoImage: {photo}")
    try:
        label.config(image=photo)
        label.image = photo  # Mantén esta línea
        print("Label updated successfully")
    except Exception as e:
        print(f"Failed to update label: {e}")


def setup_window(board):
    # Verificar si ya existe una instancia de QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("Visualizador de Ajedrez FEN")
    window.resize(400, 300)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    label = QLabel()
    layout.addWidget(label)

    update_board(board, label)

    window.show()
    app.exec_()


def correct_fen(fen):
    corrected_fen = []
    for rank in fen.split('/'):
        corrected_rank = []
        empty_count = 0
        for char in rank:
            if char.isdigit():
                empty_count += int(char)
            else:
                if empty_count > 0:
                    corrected_rank.append(str(empty_count))
                    empty_count = 0
                corrected_rank.append(char)
        if empty_count > 0:
            corrected_rank.append(str(empty_count))
        corrected_fen.append(''.join(corrected_rank))
    return '/'.join(corrected_fen)


def transform_bbox_with_matrix(bbox, M, img, width, height):
    # Definir los puntos de la bounding box como coordenadas (x, y)
    points = np.array([
        [bbox[0], bbox[1]],  # Top-left corner
        [bbox[2], bbox[1]],  # Top-right corner
        [bbox[2], bbox[3]],  # Bottom-right corner
        [bbox[0], bbox[3]]  # Bottom-left corner
    ], dtype="float32")

    # Transformar puntos
    # Cambio importante: Eliminar el tercer componente '1' y ajustar la forma de los puntos
    transformed_points = cv2.perspectiveTransform(np.array([points]), M)[0]

    # Calcular las nuevas coordenadas mínimas y máximas
    x_min, y_min = np.min(transformed_points, axis=0)
    x_max, y_max = np.max(transformed_points, axis=0)

    x_min, y_min = limit_coordinates(x_min, y_min, width, height)
    y_max, y_max = limit_coordinates(x_max, y_max, width, height)

    # plt.imshow(img)
    # Dibujar caja original en azul
    plt.plot([bbox[0], bbox[2]], [bbox[1], bbox[1]], 'b-')  # Arriba
    plt.plot([bbox[0], bbox[2]], [bbox[3], bbox[3]], 'b-')  # Abajo
    plt.plot([bbox[0], bbox[0]], [bbox[1], bbox[3]], 'b-')  # Izquierda
    plt.plot([bbox[2], bbox[2]], [bbox[1], bbox[3]], 'b-')  # Derecha

    # Dibujar caja transformada en rojo
    plt.plot([x_min, x_max], [y_min, y_min], 'r-')  # Arriba
    plt.plot([x_min, x_max], [y_max, y_max], 'r-')  # Abajo
    plt.plot([x_min, x_min], [y_min, y_max], 'r-')  # Izquierda
    plt.plot([x_max, x_max], [y_min, y_max], 'r-')  # Derecha

    # plt.show()

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def zoom_image(img_np, zoom_factor, centro=None):
    if img_np is None:
        raise ValueError("No se proporcionaron datos de imagen.")

    altura, anchura = img_np.shape[:2]

    # Si no se especifica un centro, usar el centro de la imagen
    if centro is None:
        centro = (anchura // 2, altura // 2)

    # Calcular el tamaño del rectángulo de recorte
    nueva_anchura = int(anchura / zoom_factor)
    nueva_altura = int(altura / zoom_factor)

    # Obtener las coordenadas del rectángulo de recorte
    x1 = max(0, centro[0] - nueva_anchura // 2)
    y1 = max(0, centro[1] - nueva_altura // 2)
    x2 = min(anchura, centro[0] + nueva_anchura // 2)
    y2 = min(altura, centro[1] + nueva_altura // 2)

    # Recortar y redimensionar la imagen
    imagen_recortada = img_np[y1:y2, x1:x2]
    imagen_ampliada = cv2.resize(imagen_recortada, (anchura, altura), interpolation=cv2.INTER_LINEAR)

    return imagen_ampliada


def draw_boxes_on_image(image_np, bboxes):
    """
    Dibuja rectángulos en la imagen basada en las coordenadas de las bounding boxes.
    Args:
    image_np (np.array): Imagen en formato de NumPy array.
    bboxes (list of lists): Lista de bounding boxes, cada una es [x_min, y_min, x_max, y_max].

    Returns:
    np.array: Imagen modificada con rectángulos dibujados.
    """
    for bbox in bboxes:
        # Convertir coordenadas a enteros
        start_point = (int(bbox[0]), int(bbox[1]))  # Punto superior izquierdo
        end_point = (int(bbox[2]), int(bbox[3]))  # Punto inferior derecho

        # Dibujar el rectángulo
        # - Color del rectángulo (B, G, R)
        # - Grosor del rectángulo
        image_np = cv2.rectangle(image_np, start_point, end_point, (255, 0, 0), 2)

    return image_np


def split_image(image_np):
    h, w = image_np.shape[:2]
    extra_cut = 500
    center_margin = 250  # Definir un margen para las secciones centrales
    vertical_center_margin = 200
    horizontal_center_margin = 200
    # Secciones originales
    top_left = image_np[0:h // 2, extra_cut:w // 2]
    top_right = image_np[0:h // 2, w // 2:w - extra_cut]
    bottom_left = image_np[h // 2:h, extra_cut:w // 2]
    bottom_right = image_np[h // 2:h, w // 2:w - extra_cut]

    # Secciones adicionales
    center_top = image_np[0:h // 2, w // 2 - horizontal_center_margin:w // 2 + center_margin]
    center_bottom = image_np[h // 2:h, w // 2 - horizontal_center_margin:w // 2 + center_margin]
    center_left = image_np[h // 2 - vertical_center_margin:h // 2 + vertical_center_margin, 0:w // 2]
    center_right = image_np[h // 2 - vertical_center_margin:h // 2 + vertical_center_margin, w // 2:w]
    center_total = image_np[h // 2 - center_margin:h // 2 + center_margin,
                            w // 2 - center_margin:w // 2 + center_margin]

    return [
        top_left, top_right, bottom_left, bottom_right,
        center_top, center_bottom, center_left, center_right, center_total
    ]


def limit_coordinates(x, y, width, height):
    # Limitar x e y para que no se salgan de la imagen
    x_limited = max(0, min(x, width - 1))
    y_limited = max(0, min(y, height - 1))
    return x_limited, y_limited


def adjust_bbox(bbox, dx, dy, zoom_factor):
    x_min, y_min, x_max, y_max = bbox
    # Aplicar zoom y ajustar según el desplazamiento
    x_min = (x_min * zoom_factor) + dx
    y_min = (y_min * zoom_factor) + dy
    x_max = (x_max * zoom_factor) + dx
    y_max = (y_max * zoom_factor) + dy
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


# Main
def main(image, filename):
    direc_path= f"/home/carlos/PycharmProjects/ReconocerEsquinas/results/{filename}"
    os.makedirs(direc_path, exist_ok=True)
    img = Image.open(image)
    img_np = np.array(img)

    secciones = split_image(img_np)

    factor_zoom = 1  # Ejemplo de factor de zoom

    todas_detecciones = []
    all_boxes = []
    all_conf =[]
    for i, seccion in enumerate(secciones):
        seccion_ampliada = zoom_image(seccion, factor_zoom)
        #plt.imshow(seccion_ampliada)
        #plt.show()
        detecciones, boxes = chess_pieces_detector(
            seccion_ampliada,i, direc_path)  # Ajusta esta función según tus necesidades de detección
        all_boxes.extend(boxes.cls.tolist())
        all_conf.extend(boxes.conf.tolist())
        if i < 4:  # Secciones originales
            dx = 500 if i % 2 == 0 else (i % 2) * img_np.shape[1] // 2
            dy = (i // 2) * img_np.shape[0] // 2
        else:  # Nuevas secciones
            if i in [4, 5]:  # Centros horizontales
                center_margin_adjusted = 200 * factor_zoom
                dx = (img_np.shape[1] // 2 - center_margin_adjusted)
                dy = 0 if i == 4 else (i // 2) * img_np.shape[0] // 2
            elif i in [6, 7]:  # Centros verticales
                center_margin_adjusted = 200 * factor_zoom
                dx = 0 if i == 6 else (i % 2) * img_np.shape[1] // 2
                dy = (img_np.shape[0] // 2 - center_margin_adjusted)
            else:  # Centro total
                center_margin_adjusted = 250 * factor_zoom
                dx = (img_np.shape[1] // 2 - center_margin_adjusted)
                dy = (img_np.shape[0] // 2 - center_margin_adjusted)

        for bbox in detecciones:
            bbox_ajustado = adjust_bbox(bbox, dx, dy, factor_zoom)
            todas_detecciones.append(bbox_ajustado)

    corners = detect_corners(img_np)
    img.show()

    img_transformed, M = four_point_transform(image, corners)
    img_transformed_np = np.array(img_transformed)
    img_transformed.show()

    width, height = img_transformed.size[:2]
    detections = [transform_bbox_with_matrix(bbox, M, img_transformed, width, height) for bbox in todas_detecciones]

    ptsT, ptsL = plot_grid_on_transformed_image(img_transformed)

    xA = ptsT[0][0]
    xB = ptsT[1][0]
    xC = ptsT[2][0]
    xD = ptsT[3][0]
    xE = ptsT[4][0]
    xF = ptsT[5][0]
    xG = ptsT[6][0]
    xH = ptsT[7][0]
    xI = ptsT[8][0]

    y9 = ptsL[0][1]
    y8 = ptsL[1][1]
    y7 = ptsL[2][1]
    y6 = ptsL[3][1]
    y5 = ptsL[4][1]
    y4 = ptsL[5][1]
    y3 = ptsL[6][1]
    y2 = ptsL[7][1]
    y1 = ptsL[8][1]

    # calculate all the squares

    a8 = np.array([[xA, y9], [xB, y9], [xB, y8], [xA, y8]])
    a7 = np.array([[xA, y8], [xB, y8], [xB, y7], [xA, y7]])
    a6 = np.array([[xA, y7], [xB, y7], [xB, y6], [xA, y6]])
    a5 = np.array([[xA, y6], [xB, y6], [xB, y5], [xA, y5]])
    a4 = np.array([[xA, y5], [xB, y5], [xB, y4], [xA, y4]])
    a3 = np.array([[xA, y4], [xB, y4], [xB, y3], [xA, y3]])
    a2 = np.array([[xA, y3], [xB, y3], [xB, y2], [xA, y2]])
    a1 = np.array([[xA, y2], [xB, y2], [xB, y1], [xA, y1]])

    b8 = np.array([[xB, y9], [xC, y9], [xC, y8], [xB, y8]])
    b7 = np.array([[xB, y8], [xC, y8], [xC, y7], [xB, y7]])
    b6 = np.array([[xB, y7], [xC, y7], [xC, y6], [xB, y6]])
    b5 = np.array([[xB, y6], [xC, y6], [xC, y5], [xB, y5]])
    b4 = np.array([[xB, y5], [xC, y5], [xC, y4], [xB, y4]])
    b3 = np.array([[xB, y4], [xC, y4], [xC, y3], [xB, y3]])
    b2 = np.array([[xB, y3], [xC, y3], [xC, y2], [xB, y2]])
    b1 = np.array([[xB, y2], [xC, y2], [xC, y1], [xB, y1]])

    c8 = np.array([[xC, y9], [xD, y9], [xD, y8], [xC, y8]])
    c7 = np.array([[xC, y8], [xD, y8], [xD, y7], [xC, y7]])
    c6 = np.array([[xC, y7], [xD, y7], [xD, y6], [xC, y6]])
    c5 = np.array([[xC, y6], [xD, y6], [xD, y5], [xC, y5]])
    c4 = np.array([[xC, y5], [xD, y5], [xD, y4], [xC, y4]])
    c3 = np.array([[xC, y4], [xD, y4], [xD, y3], [xC, y3]])
    c2 = np.array([[xC, y3], [xD, y3], [xD, y2], [xC, y2]])
    c1 = np.array([[xC, y2], [xD, y2], [xD, y1], [xC, y1]])

    d8 = np.array([[xD, y9], [xE, y9], [xE, y8], [xD, y8]])
    d7 = np.array([[xD, y8], [xE, y8], [xE, y7], [xD, y7]])
    d6 = np.array([[xD, y7], [xE, y7], [xE, y6], [xD, y6]])
    d5 = np.array([[xD, y6], [xE, y6], [xE, y5], [xD, y5]])
    d4 = np.array([[xD, y5], [xE, y5], [xE, y4], [xD, y4]])
    d3 = np.array([[xD, y4], [xE, y4], [xE, y3], [xD, y3]])
    d2 = np.array([[xD, y3], [xE, y3], [xE, y2], [xD, y2]])
    d1 = np.array([[xD, y2], [xE, y2], [xE, y1], [xD, y1]])

    e8 = np.array([[xE, y9], [xF, y9], [xF, y8], [xE, y8]])
    e7 = np.array([[xE, y8], [xF, y8], [xF, y7], [xE, y7]])
    e6 = np.array([[xE, y7], [xF, y7], [xF, y6], [xE, y6]])
    e5 = np.array([[xE, y6], [xF, y6], [xF, y5], [xE, y5]])
    e4 = np.array([[xE, y5], [xF, y5], [xF, y4], [xE, y4]])
    e3 = np.array([[xE, y4], [xF, y4], [xF, y3], [xE, y3]])
    e2 = np.array([[xE, y3], [xF, y3], [xF, y2], [xE, y2]])
    e1 = np.array([[xE, y2], [xF, y2], [xF, y1], [xE, y1]])

    f8 = np.array([[xF, y9], [xG, y9], [xG, y8], [xF, y8]])
    f7 = np.array([[xF, y8], [xG, y8], [xG, y7], [xF, y7]])
    f6 = np.array([[xF, y7], [xG, y7], [xG, y6], [xF, y6]])
    f5 = np.array([[xF, y6], [xG, y6], [xG, y5], [xF, y5]])
    f4 = np.array([[xF, y5], [xG, y5], [xG, y4], [xF, y4]])
    f3 = np.array([[xF, y4], [xG, y4], [xG, y3], [xF, y3]])
    f2 = np.array([[xF, y3], [xG, y3], [xG, y2], [xF, y2]])
    f1 = np.array([[xF, y2], [xG, y2], [xG, y1], [xF, y1]])

    g8 = np.array([[xG, y9], [xH, y9], [xH, y8], [xG, y8]])
    g7 = np.array([[xG, y8], [xH, y8], [xH, y7], [xG, y7]])
    g6 = np.array([[xG, y7], [xH, y7], [xH, y6], [xG, y6]])
    g5 = np.array([[xG, y6], [xH, y6], [xH, y5], [xG, y5]])
    g4 = np.array([[xG, y5], [xH, y5], [xH, y4], [xG, y4]])
    g3 = np.array([[xG, y4], [xH, y4], [xH, y3], [xG, y3]])
    g2 = np.array([[xG, y3], [xH, y3], [xH, y2], [xG, y2]])
    g1 = np.array([[xG, y2], [xH, y2], [xH, y1], [xG, y1]])

    h8 = np.array([[xH, y9], [xI, y9], [xI, y8], [xH, y8]])
    h7 = np.array([[xH, y8], [xI, y8], [xI, y7], [xH, y7]])
    h6 = np.array([[xH, y7], [xI, y7], [xI, y6], [xH, y6]])
    h5 = np.array([[xH, y6], [xI, y6], [xI, y5], [xH, y5]])
    h4 = np.array([[xH, y5], [xI, y5], [xI, y4], [xH, y4]])
    h3 = np.array([[xH, y4], [xI, y4], [xI, y3], [xH, y3]])
    h2 = np.array([[xH, y3], [xI, y3], [xI, y2], [xH, y2]])
    h1 = np.array([[xH, y2], [xI, y2], [xI, y1], [xH, y1]])

    # transforms the squares to write FEN

    FEN_annotation = [[a8, b8, c8, d8, e8, f8, g8, h8],
                      [a7, b7, c7, d7, e7, f7, g7, h7],
                      [a6, b6, c6, d6, e6, f6, g6, h6],
                      [a5, b5, c5, d5, e5, f5, g5, h5],
                      [a4, b4, c4, d4, e4, f4, g4, h4],
                      [a3, b3, c3, d3, e3, f3, g3, h3],
                      [a2, b2, c2, d2, e2, f2, g2, h2],
                      [a1, b1, c1, d1, e1, f1, g1, h1]]

    board_FEN = []
    corrected_FEN = []
    complete_board_FEN = []

    for line in FEN_annotation:
        line_to_FEN = []
        for square in line:
            piece_on_square = connect_square_to_detection(detections, square, all_boxes)
            line_to_FEN.append(piece_on_square)
        corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
        print(corrected_FEN)
        board_FEN.append(corrected_FEN)

    complete_board_FEN = [''.join(line) for line in board_FEN]

    to_FEN = '/'.join(complete_board_FEN)

    print("https://lichess.org/analysis/" + to_FEN)

    fen = correct_fen(to_FEN)

    board = chess.Board(fen)

    setup_window(board)