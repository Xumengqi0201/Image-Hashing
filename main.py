# encoding:utf-8
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import coefficient
import features
import seaborn as sns
import cv2


S = 512
level = 128
blocksize = [16, 32, 64, 128]
Thresholds = np.arange(0,1,0.01)

Attacks = {"Gamma_correction": [0.75, 0.9, 1.1, 1.25],
           "Gauss_filtering": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
           "Image_Scaling": [0.5, 0.75, 0.9, 1.1, 1.5, 2],
           "JPEG_compression": [30, 40, 50, 60, 70, 80, 90, 100],
           "Image_Rotation": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
           "Salt_and_Pepper_noise": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
           "Speckle_noise": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
           "Watermark": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           "Contrast_adjustment": [-20, -10, 10, 20],
           "Brightness_adjustment": [-20, -10, 10, 20],
           }

ParamName = {"Gamma_correction": "gamma",
             "Gauss_filtering": "Standard_deviation",
             "Image_Scaling": "Ratio",
             "JPEG_compression": "Quality_factor",
             "Image_Rotation": "Rotation_angle",
             "Salt_and_Pepper_noise": "Density",
             "Speckle_noise": "Variance",
             "Watermark": "Opacity",
             "Contrast_adjustment":"Photoshop_scale",
             "Brightness_adjustment":"Photoshop_scale",
             }


def Gethashes(imgpath, taskid, s):
    # s is block size
    # taskid is set for multiple process
    print(taskid)
    print(imgpath)
    return features.Image(imgpath, S, s, level).hashes, taskid


if __name__ == "__main__":
    sCounts = len(blocksize)
    #Thresholds = np.arange(0, 1, 0.001)
    #ThresholdCounts = len(Thresholds)
    Root = os.getcwd().replace("code", "")
    
    target = "/home/xumengqi/Downloads/multimedia/databases/Copydays/Watermark/"
    src = "/home/xumengqi/Downloads/multimedia/databases/Copydays/10/"
    name = "Opacity_10.jpg"
    files = os.listdir(src)
    
    for file in files:
        dir = file.replace("_结果.jpg", "")
        print(dir)
        #print("mv " + src + file + " " + target + dir + "/" + name)
        os.system("cp "+src+dir+"_结果.jpg" +" "+target+dir+"/"+name)

     part1 :Standard benchmark image databases
    s = 64
    SRoot = Root + "databases/Standard benchmark image/"
    # calculate hashes of original images
    labels = []
    imagelists = os.listdir(SRoot+"ORIGINAL_IMAGES/")
    imageCounts = len(imagelists)
    S_HASHES = np.zeros((imageCounts, 80), dtype=np.int)
    # labels are set for show figures
    for i in range(imageCounts):
        labels.append(imagelists[i].replace(".png", ""))
    
    # to get hashes of all original images is to avoid reduplicate calculation in the next steps
    p = ProcessPoolExecutor()  # must be used in main
    tasks = []
    for i in range(imageCounts):
        print(SRoot+"ORIGINAL_IMAGES/"+imagelists[i])
        tasks.append(p.submit(Gethashes, SRoot+"ORIGINAL_IMAGES/"+imagelists[i], i, s))
    # until all tasks are completed
    for task in as_completed(tasks):
        hashes, taskid = task.result()
        S_HASHES[taskid] = hashes
    # close ProcessPool
    p.shutdown()
    
    # get correlation of attacked images and original images
    folders = os.listdir(SRoot)
    
    for folder in folders:
        # folder is one attack kind such as Gamma_correction
        if folder == "ORIGINAL_IMAGES":
            continue
        if folder != "Watermark":
            continue
    
        params = Attacks[folder]
        print(params)
        paramCounts = len(params)
        y = np.zeros((imageCounts, paramCounts), dtype=np.float)
        for i in range(imageCounts):
            # for each image
            label = labels[i]
            p = ProcessPoolExecutor()
            tasks = []
            for j in range(paramCounts):
                param = params[j]
                pathname = SRoot + folder + "/" + label + "/" + ParamName[folder] + "_" + str(param)
                if folder == "JPEG_compression":
                    pathname = pathname + ".jpg"
                else:
                    pathname = pathname + ".png"
                print(pathname)
                tasks.append(p.submit(Gethashes, pathname, j, s))
            for task in as_completed(tasks):
                hashes, j = task.result()
                y[i][j] = coefficient.Correlation(S_HASHES[i], hashes)
            p.shutdown()
    
        # plot for each attack
        plt.figure(figsize=(8, 4))
        print(params)
        l0, = plt.plot(params, y[0], marker='o')
        l1, = plt.plot(params, y[1], marker='*')
        l2, = plt.plot(params, y[2], marker='v')
        l3, = plt.plot(params, y[3], marker='^')
        if folder == "Image_Rotation":
            plt.ylim((0.55, 1.05))
        else:
            plt.ylim((0.8, 1.05))
        plt.legend(handles=[l0, l1, l2, l3], labels=labels, loc='best')
        plt.xlabel(ParamName[folder])
        plt.ylabel("Correlation coefficient")
        plt.title(folder)
        plt.savefig(SRoot + folder + ".png")



    # part2: Copydays databases
    CRoot = Root + "databases/Copydays/"
    imagelists = os.listdir(CRoot+"ORIGINAL_IMAGES/")
    folders = os.listdir(CRoot)
    imageCounts = len(imagelists)
    pairs = imageCounts*56
    labels = []
    for i in range(imageCounts):
        labels.append(imagelists[i].replace(".jpg", ""))
    
    files = ["correlations_16.txt", "correlations_32.txt", "correlations_64.txt", "correlations_128.txt"]
    for file in files:
        os.mknod(CRoot+file)
    os.mknod(CRoot + "correlation_info.txt")
    f = open(CRoot + "correlation_info.txt", 'wt')
    
    for sid in range(sCounts):
        s = blocksize[sid]
        if s == 16:
            continue
        print("\ns:", s, file=f)
    
        hashlen = int((S/s)**2+16)  # N+16
        C_HASHES = np.zeros((imageCounts, hashlen), dtype=np.int)
        p = ProcessPoolExecutor()
        tasks = []
        for i in range(imageCounts):
            tasks.append(p.submit(Gethashes, CRoot+"ORIGINAL_IMAGES/"+imagelists[i], i, s))
        # until all tasks are completed
        for task in as_completed(tasks):
            ihashes, taskid = task.result()
            C_HASHES[taskid] = ihashes
        # close ProcessPool
        p.shutdown()
    
        alldata = np.zeros(1570)
        index = 0
        for folder in folders:
            if folder == "ORIGINAL_IMAGES":
                continue
            if folder != "Watermark":
                continue
    
            params = Attacks[folder]
            paramCounts = len(params)
    
            maxvalue = -1
            minvalue = 1
            total = 0
    
            for i in range(imageCounts):
                # for each image
                label = labels[i]
                p = ProcessPoolExecutor()
                tasks = []
                for j in range(paramCounts):
                    param = params[j]
                    pathname = CRoot + folder + "/" + label + "/" + ParamName[folder] + "_" + str(param)+".jpg"
                    tasks.append(p.submit(Gethashes, pathname, j, s))
                for task in as_completed(tasks):
                    hashes, _ = task.result()
                    r = np.corrcoef(C_HASHES[i], hashes)[0][1]
                    # print(r, file=f1)
                    alldata[index] = r
                    index += 1
                    if r > maxvalue:
                        maxvalue = r
                    if r < minvalue:
                        minvalue = r
                    total += r
                p.shutdown()
    
            meanvalue = total / (imageCounts * paramCounts)
            print(folder, file=f)
            print("max:", maxvalue, " min:", minvalue, " mean:", meanvalue, "\n", file=f)
    
        f1 = open(CRoot + files[sid], 'wt')
        for i in range(1570):
            print(alldata[i], file=f1)
        f1.close()
    
    f.close()
    #
    #



    # # part3 UCID databases
    # 887 images in UCID, total 887*886/2 = 392941 pairs
    URoot = Root + "databases/UCID/"
    imagelists = os.listdir(URoot)
    imageCounts = len(imagelists)
    pairs = int(imageCounts * (imageCounts - 1) / 2)
    
    files = ["correlations_16.txt", "correlations_32.txt", "correlations_64.txt", "correlations_128.txt"]
    for file in files:
        os.mknod(URoot + file)
    os.mknod(URoot+"correlations_info.txt")
    f = open(URoot+"correlations_info.txt", 'wt')
    for sid in range(sCounts):
        s = blocksize[sid]
        print("\ns:", s, file=f)
        hashlen = int((S/s)**2 + 16)  # N+16
        U_HASHES = np.zeros((imageCounts, hashlen), dtype=np.int)
    
        p = ProcessPoolExecutor()
        tasks = []
        for i in range(imageCounts):
            tasks.append(p.submit(Gethashes, URoot+imagelists[i], i, s))
        for task in as_completed(tasks):
            ihashes, taskid = task.result()
            U_HASHES[taskid] = ihashes
        p.shutdown(wait=True)
    
        alldata = np.zeros(pairs, dtype=np.float)
        index = 0
        maxvalue = -1
        minvalue = 1
        total = 0
        for i in range(imageCounts-1):
            for j in range(i+1, imageCounts):
                print("i:",i," j:",j)
                r = coefficient.Correlation(U_HASHES[i], U_HASHES[j])
                alldata[index] = r
                index += 1
                if r > maxvalue:
                    maxvalue = r
                if r < minvalue:
                    minvalue = r
                total += r
    
        f1 = open(URoot + files[sid], 'wt')
        for i in range(pairs):
            print(alldata[i], file=f1)
        f1.close()
    
        print("maxvalue:", maxvalue, " minvalue:", minvalue, " meanvalue:", total/pairs, file=f)
    
    
        plt.figure(figsize=(8, 4))
        sns.set_style("whitegrid")
        sns.distplot(alldata, rug=True)
        plt.xlabel("Correlation coefficient")
        plt.ylabel("probability")
        plt.title("UCID  " + "s = " + str(s))
        plt.savefig(URoot + "s_"+str(s)+".png")
    
    f.close()




    # ROC curve
    Thresholds = np.arange(0,1, 0.01)
    ThresholdCounts = len(Thresholds)
    files = ["correlations_16.txt", "correlations_32.txt", "correlations_64.txt", "correlations_128.txt"]
    TPR_matrix = np.zeros((4, ThresholdCounts))
    FPR_matrix = np.zeros((4, ThresholdCounts))
    URoot = "/home/xumengqi/Downloads/multimedia/databases/data/UcidQ/"
    CRoot = "/home/xumengqi/Downloads/multimedia/databases/data/CopydaysQ/"
    for i in range(4):
        f1 = open(CRoot+files[i], 'r')
        f2 = open(URoot+files[i], 'r')
        for line in f1:
            r1 = float(line)
            for j in range(ThresholdCounts):
                threshold = Thresholds[j]
                if r1 > threshold:
                    TPR_matrix[i][j] += 1
        for line in f2:
            r2 = float(line)
            for j in range(ThresholdCounts):
                threshold = Thresholds[j]
                if r2 > threshold:
                    FPR_matrix[i][j] += 1

        for j in range(ThresholdCounts):
            FPR_matrix[i][j] = FPR_matrix[i][j]/392055
            TPR_matrix[i][j] = TPR_matrix[i][j]/11618

    print(TPR_matrix)
    print(FPR_matrix)

    plt.figure(figsize=(8, 4))
    labels =["16×16", "32×32", "64×64", "128×128"]
    l0, = plt.plot(FPR_matrix[0], TPR_matrix[0], marker='o')
    l1, = plt.plot(FPR_matrix[1], TPR_matrix[1], marker='*')
    l2, = plt.plot(FPR_matrix[2], TPR_matrix[2], marker='v')
    l3, = plt.plot(FPR_matrix[3], TPR_matrix[3], marker='^')

    plt.legend(handles=[l0, l1, l2, l3], labels=labels, loc='best')
    plt.xlim(0, 0.1)
    plt.ylim(0.8, 0.9)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(Root + "roc.png")























