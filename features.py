# encoding:utf-8
import cv2
import numpy as np
import math

class Image(object):
    def __init__(self, imgpath, S, s, level):
        self.S = S
        self.s = s
        self.level = level

        intensity = self.Preprocess(imgpath)
        scale_intensity = self.Grayscale(intensity)
        T = self.extract_GlobalFeatures(scale_intensity)
        D = self.extract_LocalFeatures(intensity)
        self.hashes = self.Quantify(T, D)



    def Preprocess(self, imgpath):
        img = cv2.imread(imgpath)
        height = img.shape[0]
        width = img.shape[1]
        # 1. if image size is not S*S then resize it to S*S
        if height != self.S or width != self.S:
            # bilinear interpolation used by default
            img = cv2.resize(img, (self.S, self.S))

        # 2. Gauss low pass filtering and mask is 3*3
        img_RGB = cv2.GaussianBlur(img, (3, 3), 0)

        # 3. calculate I in HSI
        intensity = np.zeros((self.S, self.S), dtype=np.int)
        for i in range(self.S):
            for j in range(self.S):
                r = int(img_RGB[i][j][0])
                g = int(img_RGB[i][j][1])
                b = int(img_RGB[i][j][2])
                intensity[i, j] = round((r + g + b) / 3)

        return intensity

    def Grayscale(self, intensity):
        scale = np.zeros((self.S, self.S), dtype=np.int)
        # c = 256 / self.level
        maxpixel = intensity.max()
        minpixel = intensity.min()
        difference = maxpixel - minpixel
        for i in range(self.S):
            for j in range(self.S):
                scale[i][j] = round((self.level-1) * (intensity[i][j]-minpixel) / difference)
                #scale[i][j] = intensity[i][j] / c
        return scale


    def GLCM_theta(self, intensity, theta):
        glcm = np.zeros((self.level, self.level), dtype=np.int)
        # calculate glcm when theta = 0°，45°，90°，135°
        # make sure glcm is symmetric
        if theta == 0:
            for i in range(self.S):
                for j in range(self.S - 1):
                    rows = intensity[i][j]
                    cols = intensity[i][j + 1]
                    glcm[rows][cols] += 1
                    #glcm[cols][rows] += 1

        elif theta == 45:
            for i in range(1, self.S):
                for j in range(self.S-1):
                    rows = intensity[i][j]
                    cols = intensity[i - 1][j + 1]
                    glcm[rows][cols] += 1
                    #glcm[cols][rows] += 1

        elif theta == 90:
            for i in range(self.S - 1):
                for j in range(self.S):
                    rows = intensity[i][j]
                    cols = intensity[i + 1][j]
                    glcm[rows][cols] += 1
                    #glcm[cols][rows] += 1

        elif theta == 135:
            for i in range(1, self.S):
                for j in range(1, self.S):
                    rows = intensity[i][j]
                    cols = intensity[i - 1][j - 1]
                    glcm[rows][cols] += 1
                    #glcm[cols][rows] += 1

        return glcm


    def get_FourStatistics(self, glcm_theta):
        total = 0
        contrast = 0
        correlation = 0
        energy = 0
        homogeneity = 0

        ui = 0
        uj = 0
        delta_i = 0
        delta_j = 0

        for i in range(self.level):
            total += sum(glcm_theta[i])

        # calculate probability matrix
        p_matrix = np.zeros((self.level, self.level), dtype=np.float)
        for i in range(self.level):
            for j in range(self.level):
                p = glcm_theta[i][j] / total
                contrast += p * (i - j) * (i - j)
                energy += p * p
                homogeneity += p / (1 + (i-j)**2)
                p_matrix[i][j] = p

        for i in range(self.level):
            for j in range(self.level):
                ui += (i+1)*p_matrix[i][j]
        for j in range(self.level):
            for i in range(self.level):
                uj += (j+1)*p_matrix[i][j]
        for i in range(self.level):
            c = (i+1-ui)**2
            for j in range(self.level):
                delta_i += c *p_matrix[i][j]
        for j in range(self.level):
            c = (j+1-uj)**2
            for i in range(self.level):
                delta_j += c * p_matrix[i][j]
        # calculate correlation
        for i in range(self.level):
            for j in range(self.level):
                correlation += (i-ui)*(j-uj)*p_matrix[i][j]
        correlation = correlation / np.sqrt(delta_j * delta_i)

        #print(contrast, " ", correlation, " ", energy, " ", homogeneity)
        return contrast, correlation, energy, homogeneity



    def extract_GlobalFeatures(self, intensity):
        T = np.zeros(16, np.float)
        thetas = [0, 45, 90, 135]
        for i in range(4):
            theta = thetas[i]
            glcm = self.GLCM_theta(intensity, theta)
            T[i*4], T[i*4+1], T[i*4+2], T[i*4+3] = self.get_FourStatistics(glcm)

        return T

    def extract_LocalFeatures(self, intensity):
        n = int(self.s / 2)
        num = self.S / self.s
        N = int(num**2)
        Q = np.zeros((self.s, N), dtype=np.float)

        # dct for each block
        for i in range(N):
            offset_a = int(i // num * self.s)
            offset_b = int(i % num * self.s)
            Bi = np.float32(intensity[offset_a:offset_a+self.s, offset_b:offset_b+self.s])
            DCT = cv2.dct(Bi)

            for v in range(n):
                Q[v][i] = DCT[0][v+1]
                Q[v+n][i] = DCT[v+1][0]


        # apply data normalization for each row in Q matrix
        for i in range(self.s):
            u = np.mean(Q[i])
            std = np.std(Q[i], ddof=1)
            for j in range(N):
                Q[i][j] = (Q[i][j] - u) / std
        #U0 is a vector for distance calculation
        U0 = np.zeros(self.s, dtype=np.float)
        for i in range(self.s):
            U0[i] = np.mean(Q[i])
        #get matrix D
        D = np.zeros(N, dtype=np.float)
        for i in range(N):
            d = 0
            for l in range(self.s):
                d += (Q[l][i] - U0[l])**2
            D[i] = np.sqrt(d)

        return D

    def Quantify(self, T, D):
        length = T.shape[0] + D.shape[0]
        H = np.zeros(length, dtype=np.int16)
        for i in range(T.shape[0]):
            H[i] = round(T[i]*10 + 0.5)
        for i in range(T.shape[0], length):
            H[i] = round(D[i-T.shape[0]]*10 + 0.5)

        return H


