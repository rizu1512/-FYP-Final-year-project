import sys
from PyQt5 import QtWidgets, QtGui
from PIL import Image ,ImageDraw,Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import *
from sklearn.metrics import confusion_matrix
import os

class LungDiseaseClassifier(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Disease Classifier")
        self.resize(400, 300)
        self.setFixedSize(695, 609)
        self.setStyleSheet("background-color: #035874;")
        
        self.upload_button = QtWidgets.QPushButton(self)
        self.upload_button.setGeometry(30, 530, 201, 31)
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setGeometry(0, 0, 0, 609)
        self.background_label.setPixmap(QPixmap("peakpx.jpg").scaled(695, 609))
        self.upload_button.setStyleSheet("QPushButton{border-radius: 10px; background-color: #DF582C;}\n"
                                          "QPushButton:hover {background-color: #7D93E0;}")
        self.upload_button.setText("Upload Image")

        # Create widgets
        self.label = QLabel(self)
        self.label.setGeometry(50, -60, 541, 561)
        ###################################################################
        # self.file_label = QtWidgets.QLabel(self)
        # self.file_label.setGeometry(30, 570, 300, 31)
        # self.file_label.setStyleSheet("color: white")

        #####################################################################
        #self.label.setPixmap(QPixmap("image.gif"))
        self.gif=QMovie("image.gif")
        self.label = QLabel('darkGray',self)
        self.label.setMovie(self.gif)
        self.gif.start()
        self.label.setAlignment(Qt.AlignCenter)
        self.model = tf.keras.models.load_model('lung_disease_classifier.h5')  
        

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(0, 50, 200, 250)
        self.image_label.hide()

        #self.select_button = QtWidgets.QPushButton("Select Image", self)
        self.upload_button.setGeometry(30, 530, 201, 31)
        self.upload_button.clicked.connect(self.classify_image)
#####################################################################
        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setGeometry(0, 280, 200, 175)
        self.result_label.setStyleSheet("background-color: Gray")
        self.result_label.hide()


    def classify_image(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image")
        if file_path:
            file_name = os.path.basename(file_path)
            self.file_label.setText(f"Selected File: {file_name}")
            #self.label.clear()
            self.result_label.clear()
            self.image_label.clear()
            self.image_label.show()
            self.result_label.show()
            img = Image.open(file_path).convert("RGBA")
            img_resized = img.resize((150, 150))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            img = load_img(file_path, target_size=(150, 150))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0
            ##################################################################################
            prediction = self.model.predict(img)
            class_index = np.argmax(prediction)
            class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA','TURBERCULOSIS']  # Modify with your class labels
           ##################################################################################
            # Confusion matrix calculation
            # prediction = self.model.predict(img)
            # class_index = np.argmax(prediction)
            # class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
            y_true = np.array([1])  # Replace with actual ground truth label
            y_pred = np.array([class_index])  # Replace with predicted label
            cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
            tp = cm[1, 1]  # True Positive
            tn = cm[0, 0] + cm[0, 2] + cm[0, 3] + cm[1, 0] + cm[1, 2] + cm[1, 3] + cm[2, 0] + cm[2, 1] + cm[2, 3] + cm[3, 0] + cm[3, 1] + cm[3, 2]  # True Negative
            fp = cm[0, 1] + cm[0, 3] + cm[1, 0] + cm[1, 2] + cm[2, 0] + cm[2, 1] + cm[3, 0] + cm[3, 2]  # False Positive
            fn = cm[1, 0] + cm[1, 2] + cm[1, 3]  # False Negative
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            misclassification = (fp+fn)/(tp+tn+fp+fn)
            f1_score = (2*tp)/(2*(tp)+fp+fn)

##################################################################################################


            predicted_class = class_labels[class_index]
            predicted_prob = prediction[0][class_index]
            accuracy = round(predicted_prob * 100,2)

            self.result_label.setText(
                f"Prediction:{predicted_class} \n"
                f"Selected File: {file_name}\n"
                f"(Accuracy: {accuracy}%) \n"
                f"Recall:{recall} \n"
                f"Precision: {precision} \n"
                f"Misclassification: {misclassification}\n"
                f"F1-Score: {f1_score}\n"
                f"True Positive (TP): {tp}\n"
                f"True Negative (TN): {tn}\n"
                f"False Positive (FP): {fp}\n"
                f"False Negative (FN): {fn}\n"

                )
            
 



            # Display the selected image
            image = Image.open(file_path)
            image = image.resize((250, 250))
            image = image.convert("RGBA")
            qimage = QtGui.QImage(
                image.tobytes(),
                image.width,
                image.height,
                QtGui.QImage.Format_RGBA8888,
            )
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LungDiseaseClassifier()
    window.show()
    sys.exit(app.exec_())
