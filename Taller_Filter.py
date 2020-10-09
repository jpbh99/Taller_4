import numpy as np
import cv2
import math
from time import time

def noise(noise_typ, image):
    if noise_typ == "gauss": # Se agrega el ruido de tipo gaussiano a la imagen
        row, col = image.shape
        mean = 0
        var = 0.002
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":    # Se agrega el ruido de s&g a la imagen
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out



class Filter: # Se define la clase
    def __init__(self):    # Se define el constructor
        path = r'C:\Users\juanp\Downloads\lena.png' # Direccion de la imagen
        Image = cv2.imread(path)    # Leer imagen

        self.Image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)   #Imagen en escala de grises

        self.lena_gauss_noisy = (noise("gauss", self.Image_gray)).astype(np.uint8)  # Imagen con ruido gaussiano
        self.lena_syp_noisy = (noise("s&p", self.Image_gray))   # Imagen con ruido s&p
        #cv2.imshow("Image", self.Image_gray)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        #cv2.imshow("Image", self.lena_gauss_noisy)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

    def Gaussiano(self,In):
        if In == "gauss":   # Se realiza el filtrado de tipo gaussiano a la imagen con ruido gaussiano
            Image_gauss = cv2.GaussianBlur(self.lena_gauss_noisy, (7, 7), 1.5, 1.5)
            #cv2.imshow("Image", Image_gauss)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_gauss)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_gauss,Stimation
        elif In == "s&p":   # Se realiza el filtrado de tipo gaussiano a la imagen con ruido s&p
            Image_syp = cv2.GaussianBlur(self.lena_syp_noisy, (7, 7), 1.5, 1.5)
            #cv2.imshow("Image", Image_syp)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_syp)
            Stimation = np.mean(Stimation)
            return  self.Image_gray,Image_syp,Stimation



    def Mediana(self,In):
        if In == "gauss":   # Se realiza el filtrado de tipo Mediana a la imagen con ruido gaussiano
            Image_gauss = cv2.medianBlur(self.lena_gauss_noisy, 7)
            #cv2.imshow("Image", Image_gauss)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_gauss)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_gauss,Stimation
        elif In == "s&p":   # Se realiza el filtrado de tipo Mediana a la imagen con ruido s&p
            Image_syp = cv2.medianBlur(self.lena_syp_noisy, 7)
            #cv2.imshow("Image", Image_syp)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_syp)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_syp, Stimation


    def Bilateral(self,In):
        if In == "gauss":   # Se realiza el filtrado de tipo bilateral a la imagen con ruido gaussiano
            Image_gauss = cv2.bilateralFilter(self.lena_gauss_noisy, 15, 25, 25)
            #cv2.imshow("Image", Image_gauss)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_gauss)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_gauss,Stimation
        elif In == "s&p":   # Se realiza el filtrado de tipo bilateral a la imagen con ruido s&p
            Image_syp = cv2.bilateralFilter(self.lena_syp_noisy, 15, 25, 25)
            #cv2.imshow("Image", Image_syp)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_syp)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_syp, Stimation


    def Nlm(self,In):
        if In == "gauss":   # Se realiza el filtrado de tipo Nlm a la imagen con ruido gaussiano
            Image_gauss = cv2.fastNlMeansDenoising(self.lena_gauss_noisy, 5, 15, 25)
            #cv2.imshow("Image", Image_gauss)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_gauss)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_gauss,Stimation
        elif In == "s&p":   # Se realiza el filtrado de tipo Nlm a la imagen con ruido s&p
            Image_syp = cv2.fastNlMeansDenoising(self.lena_syp_noisy, 5, 15, 25)
            #cv2.imshow("Image", Image_syp)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            Stimation = abs(self.lena_gauss_noisy - Image_syp)
            Stimation = np.mean(Stimation)
            return self.Image_gray,Image_syp, Stimation


    def ECM(self,Image,Image_Noise):
        x = 0
        row, col = Image.shape
        for i in range(row):    # Se realiza un doble for en la que se restan los datos posicion a posicion de cada iamgen
            for j in range(col):
                x += math.pow(abs(Image_Noise[i][j] - Image[i][j]) , 2)

        Emc = math.sqrt(x/(row * col))



        print(Emc)





if __name__ == '__main__':
    Filtro = Filter()
    start_time = time() # Se toma el riempo
    Imagen,Image_Noise, Stimation = Filtro.Nlm("gauss")
    elapsed_time = time() - start_time # Se resta el tiempo actual menos el anterior

    print(elapsed_time)
    Filtro.ECM(Imagen,Image_Noise)

