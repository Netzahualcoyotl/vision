#!/usr/bin/env python
# -*- coding: latin-1 -*-
import cv2
import numpy as np


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
    
def incisoa():
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
    print("Ejecutando inciso b Cargando una imagen a partir de un archivo, cargando archivo baboon.jpg")
    imagen = cv2.imread('baboon.jpg')
    titulo = "Imagen de baboon.jpg"
    cv2.imshow(titulo,imagen)
    cv2.waitKey(10000)
    del imagen
    print("Fin del inciso b") 
    return 1

def incisoc():
    clr()
    print("Ejecutando inciso c Creando una copia de una imagen copia de baboon.jpg ")
    imagen = cv2.imread('baboon.jpg')
    imagencopia = imagen.copy()
    imagencopiamodificada = imagencopia
    imagencopiamodificada[128:256, 0:256] = np.ones((128, 256, 3), dtype='uint8')*255
    titulo1 = "Imagen baboon.jpg original"
    titulo2 = "Imagen babobon.jpg copia"
    titulo3 = "Imagen baboon.jpg copia modificada"
    cv2.imshow(titulo1,imagen)
    cv2.imshow(titulo2, imagencopia)
    cv2.imshow(titulo3, imagencopiamodificada)
    cv2.waitKey(10000)
    del imagen
    print("Fin del inciso c")
    return 1

def incisod():
    clr()
    print("Ejecutando inciso d, Separando los canales de color de una imagen")
    imagen = cv2.imread('baboon.jpg')
    tituloR = "baboon R"
    tituloG = "baboon G"
    tituloB = "baboon B"
    b, g, r = cv2.split(imagen)
    cv2.imshow(tituloR, b)
    cv2.imshow(tituloG, g)
    cv2.imshow(tituloB, r)
    cv2.waitKey(10000)
    del imagen
    print("Fin del inciso d") 
    return 1

def incisoe():
    clr()
    print("Ejecutando inciso e, Definiendo la región de interés de la imagen boboon.jpg copiada a una imagen mas pequeña")
    imagen = cv2.imread('baboon.jpg')
    imagen_region_de_interes = imagen[128:256, 0:256]
    titulo = 'Imagen baboon.jpg region de interes'
    cv2.imshow(titulo, imagen_region_de_interes)
    cv2.waitKey(10000)
    del imagen
    print("Fin del inciso e")
    return 1

def incisof():
    clr()
    print("Ejecutando inciso f, Aplicando operaciones lógicas usando dos imagenes diferentes del mismo tamaño (AND, OR, NOT, XOR)")
    imagen = np.zeros((512, 512, 3), dtype = 'uint8')
    imagen[128:256, 0:256] = np.ones((128, 256, 3), dtype = 'uint8')*255
    imagenop = imagen.copy()
    imagen1 = imagen.copy()
    imagen2 = imagen.copy()
    imagen3 = imagen.copy()
    imagen4 = imagen.copy()
    imagen2 = cv2.imread('baboon.jpg')
    imagen_and = cv2.bitwise_and(imagen2, imagen1)
    imagen_or = cv2.bitwise_or(imagen2, imagen2)
    imagen_not = cv2.bitwise_not(imagen2, imagen3)
    imagen_xor = cv2.bitwise_xor(imagen2, imagen4)
    titulo_and = "Imagen con operación AND"
    titulo_or = "Imagen con operación OR"
    titulo_not = "Imagen con operación NOT"
    titulo_xor = "Imagen con operación XOR"
    titulo = "Imagen baboon.jpg original"
    titulo2 = "imagen 512X512 blanco de alto = 128 ancho = 256 operador"
    cv2.imshow(titulo2, imagenop)
    cv2.imshow(titulo, imagen2)
    cv2.imshow(titulo_and, imagen_and)
    cv2.imshow(titulo_or, imagen_or)
    cv2.imshow(titulo_not, imagen_not)
    cv2.imshow(titulo_xor, imagen_xor)
    cv2.waitKey(80000)
    del imagen 
    del imagen1 
    del imagen2 
    del imagen3 
    del imagen4
    print("Fin del inciso f)")
    return 1

def incisog():
    print("Ejecutando opcion g), convirtiendo imagen a escala de grises")
    imagen = cv2.imread('baboon.jpg')
    imagen2 = imagen.copy()
    imagen_en_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_en_hsv = cv2.cvtColor(imagen2, cv2.COLOR_BGR2HSV)
    titulo = "Imagen baboon.jpg en grises"
    titulo2 = "Imagen baboon.jpg en HSV"
    cv2.imshow(titulo, imagen_en_grises)
    cv2.imshow(titulo2, imagen_en_hsv)
    cv2.waitKey(8000)
    print("Fin de opcion g")
    return 1

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
    
def opcion1():
    print("\t a) Crear una imagen con un cierto largo y ancho")
    print("\t b) Cargar una imagen a partir de un archivo en disco")
    print("\t c) Crear una copia de una imagen")
    print("\t d) Separar los canales de color de una imagen")
    print("\t e) Definir la región de interés de una imagen y copiarla a una imagen más pequena")
    print("\t f) Aplicar operaciones lógicas usando dos imagenes diferentes del mismo tamano (AND, OR, NOT, XOR)")
    print("\t g) Convertir la imagen a escala de grises y HSV. Mostrar cada canal por separado en la imagen HSV")
    print("\t r) Regresar")

def opciones1234():
    print("1. Utilizando las funciones de OpenCV:\n")
    print("2. Programando a mano: \n")
    print("\t a) Escribir una función que genere una imagen en espacio de color RGB normalizado a partir de una imagen en RGB")
    print("3. Uso de cámara:\n")
    print("4. Salir\n")

def clr():
    print("\033[2J\033[1;1f")
        
def regresar():
    print("Ejecutando inciso 3")
    print("Abriendo camara")
    print("Fin de inciso 3")
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
                opcion1()
                opcion_2 = raw_input()
                print(opcion_2)
                try:
                    opcion_1 = escojer[opcion_2]()
                    cv2.destroyAllWindows()
                except LookupError:
                    pass
            while opcion_1 == 2:
                try:
                    opcion_1 = opcion2()
                except LookupError:
                    pass
                
            while opcion_1 == 3:
                try:
                    opcion_1 = opcion3()
                except LookupError:
                    pass
            
            opciones1234()
        except LookupError:
            pass
