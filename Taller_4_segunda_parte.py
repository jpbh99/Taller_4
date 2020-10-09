import cv2
import numpy as np
import os
import copy
import math




Press = False
Punto = 0


def on_mouse(event, x, y, flags, param): #Interrupcion de mouse
    global Punto
    global Press
    if event == cv2.EVENT_LBUTTONDOWN:
        Punto = x, y
        Press = True

class Trans_Geo:

    def __init__(self,path1,path2):    # Se define el constructor
        self.path_file1 = path1   # Se almacena la direccion de la ruta
        self.path_file2 = path2  # Se almacena la direccion de la ruta

    def Leer(self):
        self.image1 = cv2.imread(self.path_file1) # Se almacena la imagen
        self.image2 = cv2.imread(self.path_file2)  # Se almacena la imagen

        return self.image1, self.image2





    def Puntos(self,Image):
        global Punto
        global Press
        cont = 0
        cv2.namedWindow("Jp")
        cv2.setMouseCallback("Jp", on_mouse)
        while (1):
            cv2.imshow("Jp", Image)
            if Press: # Se detecta si se presiona el boton izquierdo tres veces
                if cont == 0:
                     Pt1 = copy.copy(Punto)
                     cont = cont + 1
                     Press = False
                elif cont == 1:
                     Pt2 = copy.copy(Punto)
                     cont = cont + 1
                     Press = False
                elif cont == 2:
                     Pt3 = copy.copy(Punto)
                     Press = False
                     break



            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        return  Pt1, Pt2, Pt3

    def Affine_Trans(self,Image, Pt_in, Pt_out): #Se realiza la transfomacion afin
        Pts1 = np.float32(Pt_in)
        Pts2 = np.float32(Pt_out)
        M_affine = cv2.getAffineTransform(Pts1, Pts2)
        image_affine = cv2.warpAffine(Image, M_affine, Image.shape[:2])

        cv2.imshow("Image", image_affine)
        cv2.waitKey(0)


    def Aprox(self,Image,Pt_in,Pt_out): # Se aproxima la transformacion similar
        Pts1 = np.float32(Pt_in)
        Pts2 = np.float32(Pt_out)

        M_affine = cv2.getAffineTransform(Pts1, Pts2) # Se obtiene la matiz afin

        tx = M_affine[0, 2]         # Se despeja los valores de la matriz similar
        #print(tx)
        ty = M_affine[1, 2]
        #print(ty)
        theta_rad = np.arcsin(M_affine[1, 0])
        #print(theta_rad)
        sx = M_affine[0, 0] / np.cos(theta_rad)
        #print(sx)
        sy = M_affine[1, 1] / np.cos(theta_rad)
        #print(sy)

        # similarity
        M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],   # Se aplica la formula para obtener la matiz similar
                            [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
        image_similarity = cv2.warpAffine(Image, M_sim, Image.shape[:2])    # Se aplica la transformacion

        cv2.imshow("Image", image_similarity)
        cv2.waitKey(0)


        X1 = sx * np.cos(theta_rad)* Pts1[0][0][0] - sy * np.sin(theta_rad)*Pts1[0][0][1] + tx  #Se obtienen los tres puntos transformados
        Y1 = sx * np.sin(theta_rad) * Pts1[0][0][0] + sy * np.cos(theta_rad)*Pts1[0][0][1] + ty

        X2 = sx * np.cos(theta_rad)* Pts1[1][0][0] - sy * np.sin(theta_rad)*Pts1[1][0][1] + tx
        Y2 = sx * np.sin(theta_rad) * Pts1[1][0][0] + sy * np.cos(theta_rad)*Pts1[1][0][1] + ty

        X3 = sx * np.cos(theta_rad)* Pts1[2][0][0] - sy * np.sin(theta_rad)*Pts1[2][0][1] + tx
        Y3 = sx * np.sin(theta_rad) * Pts1[2][0][0] + sy * np.cos(theta_rad)*Pts1[2][0][1] + ty

        Error1 = math.sqrt(pow(Pts2[0][0][0] - X1, 2) + pow(Pts2[0][0][1] - Y1, 2)) #Se obtiene la norma de los tres puntos
        Error2 = math.sqrt(pow(Pts2[1][0][0] - X2, 2) + pow(Pts2[1][0][1] - Y2, 2))
        Error3 = math.sqrt(pow(Pts2[2][0][0] - X3, 2) + pow(Pts2[2][0][1] - Y3, 2))

        Error_prom = (Error1 + Error2 + Error3) / 3 # Se promedia la norma

        print("El error es de: " + str(Error_prom))



        # sx = math.sqrt(math.pow(Final_Matrix[0][0],2) + math.pow(Final_Matrix[1][0],2))
        # sy = math.sqrt(math.pow(Final_Matrix[0][1],2) + math.pow(Final_Matrix[1][1],2))
        # theta_rad = -np.arctan(Final_Matrix[1][0]*Final_Matrix[0][0])
        # tx = ((Final_Matrix[0][2] * np.cos(theta_rad)) - (Final_Matrix[1][2] * np.sin(theta_rad)))/ sx
        # ty = ((Final_Matrix[0][2] * np.sin(theta_rad)) + (Final_Matrix[1][2] * np.cos(theta_rad))) / sy
        #
        # M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
        #                      [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
        #
        # image_similarity = cv2.warpAffine(Image, M_sim, Image.shape[:2])
        # cv2.imshow("Image", image_similarity)
        # cv2.waitKey(0)



if __name__ == '__main__':
    Path = r'C:\Users\juanp\Downloads\lena.png'
    Path_ = r'C:\Users\juanp\Downloads\lena_warped.png'
    A = Trans_Geo(Path,Path_)
    Image1,Image2 = A.Leer()
    Pt1, Pt2, Pt3 = A.Puntos(Image1)
    Pt1_, Pt2_, Pt3_ = A.Puntos(Image2)

    Pt_in = [[Pt1], [Pt2], [Pt3]]
    Pt_out = [[Pt1_], [Pt2_], [Pt3_]]

    A.Affine_Trans(Image1, Pt_in, Pt_out)
    A.Aprox(Image1, Pt_in, Pt_out)
