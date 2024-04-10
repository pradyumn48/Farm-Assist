from __future__ import print_function
from tkinter import *
import tkinter
from PIL import Image, ImageTk
import tkinter.filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tkinter import messagebox
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import cv2
import h5py
import glob
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score
from tkinter import filedialog


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()  #object of tkinter library
    top = CropHome(root)  #user defined class object
    root.mainloop()  #user interface initiate

class CropHome:
    def __init__(self, top=None):#constructor function
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#d9d9d9' # X11 color: 'gray85'
        font11 = "-family {MV Boli} -size 15 -weight bold -slant roman"  \
            " -underline 0 -overstrike 0"
        font12 = "-family {Monotype Corsiva} -size 11 -weight bold "  \
            "-slant italic -underline 0 -overstrike 0"
        font9 = "-family {Segoe UI} -size 13 -weight bold -slant roman"  \
            " -underline 0 -overstrike 0"

        self.images_per_class = 800
        self.fixed_size = tuple((500, 500))
        self.train_path = "dataset/train"
        self.h5_train_data = 'model/train_data.h5'
        self.h5_train_labels = 'model/train_labels.h5'
        self.bins = 8

        self.top=top
        window_height = 818
        window_width = 1299
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        top.title("Crop Disease Prediction.")
        top.configure(background="#808080")

        img = Image.open("images/bgcrop.jpg")
        self._img1 = ImageTk.PhotoImage(img)
        background = tkinter.Label(top, image=self._img1, bd=0)
        background.pack(fill='both')
        background.image = self._img1

        self.Labelframe1 = LabelFrame(top)
        self.Labelframe1.place(relx=0.08, rely=0.13, relheight=0.35
                , relwidth=0.37)
        self.Labelframe1.configure(relief=RAISED)
        self.Labelframe1.configure(borderwidth="3")
        self.Labelframe1.configure(font=font9)
        self.Labelframe1.configure(foreground="#000080")
        self.Labelframe1.configure(relief=RAISED)
        self.Labelframe1.configure(text='''Preprocessing & Modeling.''')
        self.Labelframe1.configure(background="#ffffff")
        self.Labelframe1.configure(width=480)


        self.Button1 = Button(self.Labelframe1)
        self.Button1.place(relx=0.06, rely=0.21, height=42, width=208)
        self.Button1.configure(activebackground="#d9d9d9")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font=font12)
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Preprocessing''')
        self.Button1.configure(width=108)
        self.Button1.configure(command=self.preprocessing)

        self.Button2 = Button(self.Labelframe1)
        self.Button2.place(relx=0.06, rely=0.54, height=42, width=208)
        self.Button2.configure(activebackground="#d9d9d9")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(font=font12)
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''Model Creation''')
        self.Button2.configure(width=208)
        self.Button2.configure(command=self.modelingword)



        self.Labelframe2 = LabelFrame(top)
        self.Labelframe2.place(relx=0.52, rely=0.13, relheight=0.35
                , relwidth=0.37)
        self.Labelframe2.configure(relief=RAISED)
        self.Labelframe2.configure(borderwidth="3")
        self.Labelframe2.configure(font=font9)
        self.Labelframe2.configure(foreground="#000080")
        self.Labelframe2.configure(relief=RAISED)
        self.Labelframe2.configure(text='''Crop Prediction.''')
        self.Labelframe2.configure(background="#ffffff")
        self.Labelframe2.configure(highlightbackground="#d9d9d9")
        self.Labelframe2.configure(highlightcolor="black")
        self.Labelframe2.configure(width=480)

        self.Entry3 = Entry(self.Labelframe2)
        self.Entry3.place(relx=0.33, rely=0.2, relheight=0.25, relwidth=0.48)
        self.Entry3.configure(background="white")
        self.Entry3.configure(disabledforeground="#a3a3a3")
        self.Entry3.configure(font="TkFixedFont")
        self.Entry3.configure(foreground="#000000")
        self.Entry3.configure(highlightbackground="#d9d9d9")
        self.Entry3.configure(highlightcolor="black")
        self.Entry3.configure(insertbackground="black")
        self.Entry3.configure(selectbackground="#c4c4c4")
        self.Entry3.configure(selectforeground="black")


        self.Button5 = Button(self.Labelframe2)
        self.Button5.place(relx=0.33, rely=0.52, height=42, width=208)
        self.Button5.configure(activebackground="#d9d9d9")
        self.Button5.configure(activeforeground="#000000")
        self.Button5.configure(background="#d9d9d9")
        self.Button5.configure(disabledforeground="#a3a3a3")
        self.Button5.configure(font=font12)
        self.Button5.configure(foreground="#000000")
        self.Button5.configure(highlightbackground="#d9d9d9")
        self.Button5.configure(highlightcolor="black")
        self.Button5.configure(pady="0")
        self.Button5.configure(text='''Prediction.''')
        self.Button5.configure(command=self.prediction)

        self.Label1 = Label(top)
        self.Label1.place(relx=0.07, rely=0.05, height=41, width=227)
        self.Label1.configure(background="#000080")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font11)
        self.Label1.configure(foreground="#ffffff")
        self.Label1.configure(text='''Crop Disease Pre...''')
        self.Label1.configure(width=227)

        self.Label5 = Label(top)
        self.Label5.place(relx=0.01, rely=0.49, height=41, width=147)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(activeforeground="black")
        self.Label5.configure(background="#000080")
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(font=font11)
        self.Label5.configure(foreground="#ffffff")
        self.Label5.configure(highlightbackground="#d9d9d9")
        self.Label5.configure(highlightcolor="black")
        self.Label5.configure(text='''Outputs.''')
        self.Label5.configure(width=147)

        self.Frame1 = Frame(top)
        self.Frame1.place(relx=0.02, rely=0.56, relheight=0.42, relwidth=0.97)
        self.Frame1.configure(relief=RAISED)
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief=RAISED)
        self.Frame1.configure(background="#808080")
        self.Frame1.configure(width=1265)

        self.Label4 = Label(self.Frame1)
        self.Label4.place(relx=0.0, rely=0.03, height=331, width=417)
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''Label''')
        self.Label4.configure(width=417)

        self.Label6 = Label(self.Frame1)
        self.Label6.place(relx=0.33, rely=0.03, height=331, width=417)
        self.Label6.configure(activebackground="#f9f9f9")
        self.Label6.configure(activeforeground="black")
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(highlightbackground="#d9d9d9")
        self.Label6.configure(highlightcolor="black")
        self.Label6.configure(text='''Label''')

        self.Label7 = Label(self.Frame1)
        self.Label7.place(relx=0.66, rely=0.03, height=331, width=417)
        self.Label7.configure(activebackground="#f9f9f9")
        self.Label7.configure(activeforeground="black")
        self.Label7.configure(background="#d9d9d9")
        self.Label7.configure(disabledforeground="#a3a3a3")
        self.Label7.configure(foreground="#000000")
        self.Label7.configure(highlightbackground="#d9d9d9")
        self.Label7.configure(highlightcolor="black")
        self.Label7.configure(text='''Label''')

    def rgb_bgr(self,image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_img

    def bgr_hsv(self,rgb_img):
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        return hsv_img

    def img_segmentation(self,rgb_img, hsv_img):
        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
        lower_brown = np.array([10, 0, 10])
        upper_brown = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
        disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
        final_mask = healthy_mask + disease_mask
        final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
        return final_result

    def fd_hu_moments(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def fd_haralick(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def fd_histogram(self,image, mask=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()



    def preprocessing(self):

        train_labels = os.listdir(self.train_path)

        # sort the training labels
        train_labels.sort()
        print(train_labels)


        self.global_features = []
        self.labels = []

        for training_name in train_labels:

            dir = os.path.join(self.train_path, training_name)


            current_label = training_name


            for x in range(1, self.images_per_class + 1):
                # get the image file name
                file = dir + "/" + str(x) + ".jpg"


                image = cv2.imread(file)
                image = cv2.resize(image, self.fixed_size)


                RGB_BGR = self.rgb_bgr(image)
                BGR_HSV = self.bgr_hsv(RGB_BGR)
                IMG_SEGMENT = self.img_segmentation(RGB_BGR, BGR_HSV)


                fv_hu_moments = self.fd_hu_moments(IMG_SEGMENT)
                fv_haralick = self.fd_haralick(IMG_SEGMENT)
                fv_histogram = self.fd_histogram(IMG_SEGMENT)

                # Concatenate
                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

                self.labels.append(current_label)
                self.global_features.append(global_feature)

            print("[STATUS] processed folder: {}".format(current_label))

        print("[STATUS] completed Global Feature Extraction...")
        print("[STATUS] feature vector size {}".format(np.array(self.global_features).shape))
        print("[STATUS] training Labels {}".format(np.array(self.labels).shape))

        targetNames = np.unique(self.labels)
        le = LabelEncoder()
        target = le.fit_transform(self.labels)
        print("[STATUS] training labels encoded...")


        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(self.global_features)
        print("[STATUS] feature vector normalized...")

        print("[STATUS] target labels: {}".format(target))
        print("[STATUS] target labels shape: {}".format(target.shape))


        h5f_data = h5py.File(self.h5_train_data, 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

        h5f_label = h5py.File(self.h5_train_labels, 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))

        h5f_data.close()
        h5f_label.close()



        messagebox.showinfo("Message", "Processed Successfully.")

    def modelingword(self):
        warnings.filterwarnings('ignore')
        self.num_trees = 100
        self.test_size = 0.20
        self.seed = 9
        self.train_path = "dataset/train"
        self.test_path = "dataset/test"
        self.h5_train_data = 'model/train_data.h5'
        self.h5_train_labels = 'model/train_labels.h5'
        self.scoring = "accuracy"

        train_labels = os.listdir(self.train_path)
        train_labels.sort()

        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        # create all the machine learning models
        self.models = []
        self.models.append(('LR', LogisticRegression(random_state=self.seed)))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier(random_state=self.seed)))
        self.models.append(('RF', RandomForestClassifier(n_estimators=self.num_trees, random_state=self.seed)))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(random_state=self.seed)))



        # variables to hold the results and names
        self.results = []
        self.names = []

        h5f_data = h5py.File(self.h5_train_data, 'r')
        h5f_label = h5py.File(self.h5_train_labels, 'r')

        global_features_string = h5f_data['dataset_1']
        global_labels_string = h5f_label['dataset_1']

        self.global_features = np.array(global_features_string)
        self.global_labels = np.array(global_labels_string)

        h5f_data.close()
        h5f_label.close()

        print("[STATUS] features shape: {}".format(self.global_features.shape))
        print("[STATUS] labels shape: {}".format(self.global_labels.shape))

        print("[STATUS] training started...")

        (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
            np.array(self.global_features),
            np.array(self.global_labels),
            test_size=self.test_size,
            random_state=self.seed)

        print("[STATUS] splitted train and test data...")
        print("Train data  : {}".format(trainDataGlobal.shape))
        print("Test data   : {}".format(testDataGlobal.shape))


        for name, model in self.models:
            kfold = KFold(n_splits=10, random_state=self.seed)
            cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=self.scoring)
            self.results.append(cv_results)
            self.names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # boxplot algorithm comparison
        fig = pyplot.figure()
        fig.suptitle('Machine Learning algorithm comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.savefig(os.path.join("model", 'graph2.png'), dpi=100)
        plt.clf()



        self.clf = RandomForestClassifier(n_estimators=self.num_trees, random_state=self.seed)
        self.clf.fit(trainDataGlobal, trainLabelsGlobal)

        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                               oob_score=False, random_state=9, verbose=0, warm_start=False)

        y_predict = self.clf.predict(testDataGlobal)


        print(testDataGlobal)

        cm = confusion_matrix(testLabelsGlobal, y_predict)

        sns.heatmap(cm, annot=True)

        print(classification_report(testLabelsGlobal, y_predict))

        acc=accuracy_score(testLabelsGlobal, y_predict)
        print(acc)
        messagebox.showinfo("Accuracy",acc)

        image = Image.open("model/graph2.png")

        image = image.resize((350, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        self.Label4.configure(image=photo)
        self.Label4.image = photo

        messagebox.showinfo("Message", "Process Completed Successfully.")


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Prediction Accuracy")
        plt.legend(loc='best')
        plt.savefig(os.path.join("output", 'prediction.png'), dpi=100)
        plt.clf()




    def prediction(self):
        global_features1 = []
        labels1 = []

        ftypes = [('CSV Files', '*.csv'), ('All files', '*')]
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        file =filename

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, self.fixed_size)

        # Running Function Bit By Bit
        RGB_BGR = self.rgb_bgr(image)
        BGR_HSV = self.bgr_hsv(RGB_BGR)
        IMG_SEGMENT = self.img_segmentation(RGB_BGR, BGR_HSV)

        # Call for Global Fetaure Descriptors
        fv_hu_moments = self.fd_hu_moments(IMG_SEGMENT)
        fv_haralick = self.fd_haralick(IMG_SEGMENT)
        fv_histogram = self.fd_histogram(IMG_SEGMENT)

        # Concatenate
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        labels1.append("test")
        global_features1.append(global_feature)

        global_features1=[global_features1]

        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # rescaled_features = scaler.fit_transform(global_features1)
        y_predict = self.clf.predict(global_features1[0])
        print(y_predict)

        # d = {"text": [datainput]}
        # self.test = pd.DataFrame(d)


        messagebox.showinfo("Message", "Processed Successfully.")

if __name__ == '__main__':
    vp_start_gui()
