#!/usr/bin/env python
# -*- coding: latin-1 -*-
import cv2
import numpy as np
import os


def a():
    img = np.zeros((512, 512, 3), dtype = 'uint8')
    img[128:256,0:256] = np.ones((128, 256,3), dtype = 'uint8')*255
    img[256, 256] = (255, 255, 255)#BRG
    cv2.imshow('Imagen_en_negro',img)
    baboon = cv2.imread('baboon.jpg')
    img[128:256,0:256] = np.ones((128, 256,3), dtype = 'uint8')*255
    img[256, 256] = (255, 255, 255)#BRG
    cv2.imshow('imagen',baboon)
    cv2.waitKey(500)    

def get_corners():
    cols = 15
    rows = 20
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2]=np.mgrid[0:rows,0:cols].T.reshape(-1, 2)
    objpoints = [] #3d
    imgpoints = [] #2d
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    files = next(os.walk("data"))[2]
    print(files)
    for f in files:
        imagen = cv2.imread("data/" + f)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)
        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(imagen, (rows, cols), refined_corners, ret)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            for i in range(0,299):
                x = refined_corners[i][0][0]
                y = refined_corners[i][0][1]
                cv2.putText(imagen,str(i),(x,y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('Original', imagen)
            cv2.waitKey(200)
    return imgpoints, objpoints, img_size

def incisoa():
    imgpoints, objpoints, img_size = get_corners()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print(mtx)
    print(dist)
    np.savez("Parámetros_de_calibración", mtx = mtx, dist = dist)
    return 1


    
def incisoaa():
    clr()
    print("Ejecutando inciso a)")
    ancho = 500
    alto = 50
    imagen = np.zeros((alto, ancho, 3), dtype = 'uint8')
    titulo = "imagen ancho = "+ str(ancho)+" alto = " + str(alto)
    cv2.imshow(titulo, imagen)
    cv2.waitKey(10000) 
    del imagen
    print("Fin del inciso a)")
    return 1
    
def incisob():
    clr()
    print("Ejecutando inciso b Cargando una imagenes a partir de archivos, cargando imagenes")
    cols = 15
    rows = 20
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2]=np.mgrid[0:rows,0:cols].T.reshape(-1, 2)
    objpoints = [] #3d
    imgpoints = [] #2d
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    files = next(os.walk("data"))[2]
    print(files)
    for f in files:
        imagen = cv2.imread("data/" + f)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)
        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(imagen, (rows, cols), refined_corners, ret)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            print(refined_corners[299][0][0],refined_corners[299][0][1])
            print(str(objp[299]))
            print("-------")
            for i in range(0,299,7):
                x = refined_corners[i][0][0]
                y = refined_corners[i][0][1]
                cv2.putText(imagen,str(i)+str(objp[i]),(x,y), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                
            cv2.imshow('Original', imagen)
            cv2.waitKey(400)
    del imagen
    print("Fin del inciso b") 
    return 1

def incisoc():
    clr()
    print("Ejecutando inciso c combirtiendo archivos a gris ")
    files = next(os.walk("data"))[2]
    print(files)
    for f in files:
        imagen = cv2.imread("data/" + f)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("data/"+f,gris)
    del imagen
    print("Fin del inciso c")
    return 1

def incisod():
    clr()
    print("Ejecutando inciso d mostrando esquinas")
    cols = 15
    rows = 20
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2]=np.mgrid[0:rows,0:cols].T.reshape(-1, 2)
    objpoints = [] #3d
    imgpoints = [] #2d
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    files = next(os.walk("data"))[2]
    print(files)
    for f in files:
        imagen = cv2.imread("data/" + f)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(imagen, (rows, cols), corners, ret)              
            cv2.imshow('Original', imagen)
            cv2.waitKey(400)
    del imagen
    print("Fin del inciso d") 
    return 1

def incisoe():
    clr()
    print("Ejecutando inciso d mostrando esquinas refinadas")
    cols = 15
    rows = 20
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2]=np.mgrid[0:rows,0:cols].T.reshape(-1, 2)
    objpoints = [] #3d
    imgpoints = [] #2d
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    files = next(os.walk("data"))[2]
    print(files)
    for f in files:
        imagen = cv2.imread("data/" + f)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)
        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(imagen, (rows, cols), refined_corners, ret)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            print(refined_corners[299][0][0],refined_corners[299][0][1])
            print(str(objp[299]))
            print("-------")
            for i in range(0,299,7):
                x = refined_corners[19][0][0]
                y = refined_corners[19][0][1]
                cv2.putText(imagen,"Esquinas refinadas",(x,y), font, 1, (255, 0, 0), 1, cv2.LINE_AA)              
            cv2.imshow('Original', imagen)
            cv2.waitKey(400)
    del imagen
    print("Fin del inciso d") 
    return 1

def incisof():
    print("Ejecutando opcion g), Obteniendo parámetros de la cámara...")
    imgpoints, objpoints, img_size = get_corners()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print(mtx)
    print(dist)
    np.savez("Parámetros_de_calibración", mtx = mtx, dist = dist)
    return 1

def incisog():
    print("Ejecutando opcion g), Obteniendo parámetros de la cámara...")
    imgpoints, objpoints, img_size = get_corners()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print(mtx)
    print(dist)
    print("Guardando parámetros")
    np.savez("Parámetros_de_calibración", mtx = mtx, dist = dist)
    return 1



def incisoa2():
    print("Ejecutando opción a) cargando parámetros y corrigiendo imagenes")
    data = np.load('Parámetros_de_calibración.npz')
    mtx = data['mtx']
    dist = data['dist']
    files = next(os.walk("data"))[2]
    print("Presione cualquier tecla")
    for f in files:
        img = cv2.imread("data/" +f)
        und = cv2.undistort(img, mtx, dist)
        cv2.imshow('Original', img)
        cv2.imshow('No distorsionada', und)
        cv2.waitKey(0) 
    return 2
def incisob2():
    print("Ejecutando opción b) cargando parámetros y corrigiendo imagenes")
    data = np.load('Parámetros_de_calibración.npz')
    mtx = data['mtx']
    dist = data['dist']
    files = next(os.walk("data"))[2]
    print("Presione cualquier tecla")
    for f in files:
        img = cv2.imread("data/" +f)
        und = cv2.undistort(img, mtx, dist)
        cv2.imshow('Original', img)
        cv2.imshow('No distorsionada', und)
        cv2.waitKey(0) 
    return 2
def incisoc2():
    print("Ejecutando opción c) Imagenes original y sin distorsión")
    data = np.load('Parámetros_de_calibración.npz')
    mtx = data['mtx']
    dist = data['dist']
    files = next(os.walk("data"))[2]
    print("Presione cualquier tecla")
    for f in files:
        img = cv2.imread("data/" +f)
        und = cv2.undistort(img, mtx, dist)
        cv2.imshow('Original', img)
        cv2.imshow('No distorsionada', und)
        cv2.waitKey(0)
    return 2



def incisoa3():
    print("Ejecutando opción a) Imprimiendo la imagen chessboard.png")
    imagen = cv2.imread("chessboard.png")
    print("Presione cualquier tecla")
    cv2.imshow("Chessboard.png", imagen)
    print("Presione cualquier tecla para terminar")
    cv2.waitKey(0) 
    return 2

    return 3
def incisob3():
    print("Ejecutando opcion b) Tomando con una WebCam, varias fotos del tablero desde varios ángulos, cuando menos diez imágenes")
    clr()
    print("Abriendo camara")
    contador = 0
    captura = cv2.VideoCapture(contador)
    while not captura.isOpened():
        captura = cv2.VideoCapture(contador)
        contador = contador +1
        if contador == 11:
            break
    if not captura.isOpened():
        print("No se puede abrir la camara")
        exit()
    print("Presione ESC para salir")
    contador = 1
    print("Introduzca el número de fotos a tomar: ")
    b = input()
    while cv2.waitKey(10) != 27 and contador <= b:
        try:
            clr()
            print("Foto numero = {} presione ESC para salir seleccionando primero una imagen".format(contador))
            os.system("mpg123 " + "camara.mp3")
            bandera, imagen = captura.read() 
            if not bandera:
                print("No se puede recibir la captura de imagenes, ¿Termino el video?. Saliendo..")
                break
            print("Mueba el tablero de posición para tomar la siguiente foto las fotos se tomarán cada segundo")
            print("Guardando imagen data/foto{}.jpg ...".format(contador))
            foto = 'data/foto{}.jpg'.format(contador)
            cv2.imwrite(foto,imagen)
            contador = contador + 1
            cv2.waitKey(1000)
        except LookupError:
            pass    
    cv2.destroyAllWindows()
    print("Fin de opción b) {} imagenes guardadas en /data\n\n\n......Ya".format(contador))
    return 3
def incisoc3():
    print("Ejecutando opción c) Obteniendo parámetros de calibración de la carpeta data/")
    imgpoints, objpoints, img_size = get_corners()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,img_size, None, None)
    print(mtx)
    print(dist)
    np.savez("Parámetros_de_calibración", mtx = mtx, dist = dist)
    return 3
def incisod3():
    print("Ejecutando opcion b) Tomando con una WebCam, varias fotos del tablero desde varios ángulos, cuando menos diez imágenes")
    clr()
    print("Abriendo camara")
    contador = 0
    captura = cv2.VideoCapture(contador)
    bandera, imagen = captura.read()
    while not captura.isOpened():
        captura = cv2.VideoCapture(contador)
        contador = contador +1
        if contador == 11:
            break
    if not captura.isOpened():
        print("No se puede abrir la camara")
        exit()
    print("Cargando parámetros de la cámara")
    data = np.load('Parámetros_de_calibración.npz')
    mtx = data['mtx']
    dist = data['dist']
    while cv2.waitKey(10) != 27:
        try:
            clr()
            print("Foto numero = {} presione ESC para salir seleccionando primero una imagen".format(contador)) 
            bandera, imagen = captura.read()
            if not bandera:
                print("No se puede recibir la captura de imagenes, ¿Termino el video?. Saliendo..")
                break
            
            cv2.imshow("Video Original",imagen)
            sindistorsion = cv2.undistort(imagen, mtx, dist)
            cv2.imshow("Video sin distorsión",sindistorsion)
            contador = contador + 1
        except LookupError:
            pass    
    cv2.destroyAllWindows()
    print("Fin de opción b) {} imagenes guardadas en /data\n\n\n......Ya".format(contador))
    return 3






















def opcion3():
    clr()
    print("Ejecutando opcion 3")
    print("Abriendo camara")
    contador = 0
    captura = cv2.VideoCapture(contador)
    while not captura.isOpened():
        captura = cv2.VideoCapture(contador)
        contador = contador +1
        if contador == 11:
            break
    if not captura.isOpened():
        print("No se puede abrir la camara")
        exit()
    print("Presione ESC para salir")
    while cv2.waitKey(10) != 27:
        try:
            clr()
            print("Foto numero = {} presione ESC para salir seleccionando primero una imagen".format(contador))
            bandera, imagen = captura.read()
            if not bandera:
                print("No se puede recibir la captura de imagenes, ¿Termino el video?. Saliendo..")
                break
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen2 = imagen.copy()
            imagen2[128:256, 0:256] = np.ones((128, 256, 3), dtype = 'uint8')*255
            imagen3 = imagen2.copy()
            imagen_and = cv2.bitwise_xor(imagen3, imagen)
            titulo = "Imagen en gris "
            titulo2 = "Imagen en color"
            titulo3 = "Operación xor"
            titulo4 = "Imagen Operador 128 x 256 negro"
            cv2.imshow(titulo, gris)
            cv2.imshow(titulo2, imagen)
            cv2.imshow(titulo3, imagen_and)
            cv2.imshow(titulo4, imagen2)
            contador = contador + 1
        except LookupError:
            pass
    cv2.destroyAllWindows()
    return 0
        
def opcion2():
    clr()
    print("Ejecutando opcion 2")
    print("Escritura de una función que genera una imagen \nEn espacio de color RGB \nnormalizado a partir de una imagen en RGB")
    def do_something(imagen):
        resultado = np.zeros(imagen.shape, dtype = 'uint8')        
        [h,w,c] = imagen.shape
        for i in range(h):
            for j in range(w):
                b,g,r = imagen[i,j]
                sum = float(b) + g + r
                if sum > 0:
                    resultado[i,j] = (255*b/sum, 255*g/sum, 255*r/sum)
        return resultado
    img_baboon = cv2.imread('baboon.jpg')
    img_normalized = do_something(img_baboon)
    cv2.imshow("Original BGR", img_baboon)
    cv2.imshow("Normalized BGR", img_normalized)
    print("Imagen normalizada")
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    print("Fin de la opcion 2")
    return 0
    
    
    
    
    

    
    
   
    
    
    
    
    
    
    
    
    
def opcion1T():
    print("\t a) Identifique el número de esquinas (rxc) que se pueden detectar en los tableros \n\t    de ajedrez de las imágenes de la carpeta data\n")
    print("\t b) Genere un conjunto de puntos P de la forma (0,0,0),...,(i,j,0)...(r,c,0). \n \t    Estos puntos representarán las coordenadas en 3D de las esquinas detectadas en un tablero. \n \t    Estas coordenadas serán las  mismas para todas las imágenes de muestra\n")
    print("\t c) Cargar todas las imágenes de muestra de la carpeta data y transformarlas a escala de grises\n")
    print("\t d) Utilizando la función findChessboardCorners, encontrar las esquinas del tablero para cada imagen.\n")
    print("\t e) Utilizando la función cornerSubPix, refinar las coordenadas de las esquinas encontradas\n")
    print("\t f) Utilizando el conjunto de puntos P y las esquinas encontradas en cada imagen, \n \t    determinar los parámetros de calibración utilizando la función calibrateCamera de OpenCV.\n")
    print("\t g) Almacenar en un archivo en disco la matriz de parámetros intrínsecos y el arreglo\n \t    con los coeficientes de distorsión. Se sugiere el uso de la función savez de la biblioteca numpy.\n")
    print("\t r) Regresar")


def opcion2T():
    print("\t a) Cargar los parámetros de calibración almacenados en disco. Se sugiere el uso \n \t    de la función load de la biblioteca numpy.\n")
    print("\t b) Corregir la distorsión en todas las imágenes de la carpeta data.\n")
    print("\t c) Mostrar en pantalla la imagen original y la imagenn sin distorsión\n")
    print("\t r) Regresar")
    
def opcion3T():
    print("\t a) Imprimir la imagen chessboard.png\n")
    print("\t b) Con una WebCam, tomar varias fotos del tablero desde varios ángulos, cuando menos diez imágenes\n")
    print("\t c) Realizar un proceso similar al inciso 1 para calibrar la cámara utilizada.\n")
    print("\t d) Hacer un programa que obtenga el video de la cámara y corrija la distorsión. \n \t    Desplegar tanto el video original como el video sin distorsión.\n")
    print("\t r) Regresar")

def opciones1234():
    print("1. Utilizando el tutorial publicado en https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html realizar lo siguiente:\n")
    print("2. Utilizando el mismo tutorial: \n")
    print("3. Calibración de una WebCam:\n")
    print("4. Salir\n")

def clr():
    print("\033[2J\033[1;1f")
        
def regresar():
    return 0
    
if __name__ == '__main__':
    
    escojer = {
                "a": incisoa,
                "b": incisob,
                "c": incisoc,
                "d": incisod,
                "e": incisoe,
                "f": incisof,
                "g": incisog,
                "r": regresar
                }
                
    escojer2 = {
                "a": incisoa2,
                "b": incisob2,
                "c": incisoc2,
                "r": regresar
                }                

    escojer3 = {
                "a": incisoa3,
                "b": incisob3,
                "c": incisoc3,
                "d": incisod3,
                "r": regresar
                }    
    print("Entrega de tarea seleccione la opción: \n") 

    opciones1234()
    print("\t\t Opción: \r")
    opcion_1 = 0
    while opcion_1 != 4:
        try:
            opcion_1 = input()
            while opcion_1 == 1:
                clr()
                print("Seleccione la opción: \n")
                opcion1T()
                opcion_2 = raw_input()
                print(opcion_2)
                try:
                    opcion_1 = escojer[opcion_2]()
                    cv2.destroyAllWindows()
                except LookupError:
                    pass
            while opcion_1 == 2:
                clr()
                print("Seleccione la opción: \n")
                opcion2T()
                opcion_2 = raw_input()
                print(opcion_2)
                try:
                    opcion_1 = escojer2[opcion_2]()
                    cv2.destroyAllWindows()
                except LookupError:
                    pass
                
            while opcion_1 == 3:
                clr()
                print("Seleccione la opción\n")
                opcion3T()
                opcion_2 = raw_input()
                print(opcion_2)
                try:
                    opcion_1 = escojer3[opcion_2]()
                    cv2.destroyAllWindows()
                except LookupError:
                    pass
            clr()
            opciones1234()
        except LookupError:
            pass
