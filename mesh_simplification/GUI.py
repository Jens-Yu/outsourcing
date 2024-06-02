from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QFileDialog, QGraphicsPathItem,QOpenGLWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QPainterPath



class Ui_control(object):
     def setupUi(self, simplify):
         print(0)

         simplify.setObjectName("simplify")
         simplify.resize(600,400)
         simplify.setMinimumSize(QtCore.QSize(600, 400))
         simplify.setMaximumSize(QtCore.QSize(600,400))
         simplify.setAnimated(True)
         self.centralwidget = QtWidgets.QWidget(simplify)
         self.centralwidget.setObjectName("centralwidget")
         #layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(self.centralwidget)
         #文字部分
         simplify.setCentralWidget(self.centralwidget)
         #self.label_import = QtWidgets.QLabel(self.centralwidget)  # 文字：模型导入
         #self.label_import.setGeometry(QtCore.QRect(100, 120, 131, 40))
         #self.label_import.setMinimumSize(QtCore.QSize(110, 40))
         #font = QtGui.QFont()
         #font.setPointSize(10)
         #self.label_import.setFont(font)
         #self.label_import.setObjectName("label_import")
         self.label_ratio=QtWidgets.QLabel(self.centralwidget)#文字：简化率
         self.label_ratio.setGeometry(QtCore.QRect(10,20,150,40))
         self.label_ratio.setMinimumSize(QtCore.QSize(200,40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_ratio.setFont(font)
         self.label_ratio.setObjectName("label_ratio")
         self.label_model1=QtWidgets.QLabel(self.centralwidget)#文字：原模型
         self.label_model1.setGeometry(QtCore.QRect(400,20, 100, 40))
         self.label_model1.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model1.setFont(font)
         self.label_model1.setObjectName("label_model1")
         self.label_model3=QtWidgets.QLabel(self.centralwidget)#文字：文件体积
         self.label_model3.setGeometry(QtCore.QRect(360,65, 100, 40))
         self.label_model3.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model3.setFont(font)
         self.label_model3.setObjectName("label_model3")
         self.label_model4=QtWidgets.QLabel(self.centralwidget)#文字：顶点数量
         self.label_model4.setGeometry(QtCore.QRect(360,105, 100, 40))
         self.label_model4.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model4.setFont(font)
         self.label_model4.setObjectName("label_model4")
         self.label_model5=QtWidgets.QLabel(self.centralwidget)#文字：面片数量
         self.label_model5.setGeometry(QtCore.QRect(360,145, 100, 40))
         self.label_model5.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model5.setFont(font)
         self.label_model5.setObjectName("label_model5")
         self.label_model2=QtWidgets.QLabel(self.centralwidget)#文字：简化后模型
         self.label_model2.setGeometry(QtCore.QRect(380, 190, 200, 40))
         self.label_model2.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model2.setFont(font)
         self.label_model2.setObjectName("label_model2")
         self.label_model6=QtWidgets.QLabel(self.centralwidget)#文字：简化后文件体积
         self.label_model6.setGeometry(QtCore.QRect(340,235, 150, 40))
         self.label_model6.setMinimumSize(QtCore.QSize(150, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model6.setFont(font)
         self.label_model6.setObjectName("label_model6")
         self.label_model7=QtWidgets.QLabel(self.centralwidget)#文字：顶点数量
         self.label_model7.setGeometry(QtCore.QRect(340,275, 150, 40))
         self.label_model7.setMinimumSize(QtCore.QSize(150, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model7.setFont(font)
         self.label_model7.setObjectName("label_model7")
         self.label_model8=QtWidgets.QLabel(self.centralwidget)#文字：面片数量
         self.label_model8.setGeometry(QtCore.QRect(340,315, 150, 40))
         self.label_model8.setMinimumSize(QtCore.QSize(150, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_model8.setFont(font)
         self.label_model8.setObjectName("label_model8")
         self.label_mode_selected=QtWidgets.QLabel(self.centralwidget)#文字：模式选择
         self.label_mode_selected.setGeometry(QtCore.QRect(10, 300, 100, 40))
         self.label_mode_selected.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.label_mode_selected.setVisible(True)
         self.label_mode_selected.setFont(font)
         self.label_mode_selected.setObjectName("label_mode_selected")
         #控件
         self.port_list = QtWidgets.QComboBox(self.centralwidget)  # 下拉框：模式选择
         self.port_list.setGeometry(QtCore.QRect(110, 300, 100, 40))
         self.port_list.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.port_list.setFont(font)
         self.port_list.setObjectName("port_list")
         self.text_edit=QtWidgets.QLineEdit(self.centralwidget)#简化率输入
         self.text_edit.setGeometry(QtCore.QRect(180, 20, 100, 40))
         self.text_edit.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.text_edit.setFont(font)
         self.text_edit.setObjectName("text_edit")
         #ratio=self.text_edit.Text()
         self.load_model = QtWidgets.QPushButton(self.centralwidget)  # 点击导入模型
         self.load_model.setGeometry(QtCore.QRect(60, 150, 100, 40))
         self.load_model.setMinimumSize(QtCore.QSize(100, 40))
         font = QtGui.QFont()
         font.setPointSize(10)
         self.load_model.setFont(font)
         self.load_model.setObjectName("load_model")


         #显示框
        # self.scene = QGraphicsScene()
         #self.model1_display = QtWidgets.QOpenGLWidget(self.centralwidget)#原模型显示
         #self.model1_display.setEnabled(True)
        # self.model1_display.setGeometry(QtCore.QRect(500, 40, 400, 400))
        # self.model1_display.setMinimumSize(QtCore.QSize(400, 400))
        # self.model1_display.setAutoFillBackground(False)
        # self.model1_display.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.model1_display.setObjectName("model1_display")


        # self.model2_display = QtWidgets.QOpenGLWidget(self.centralwidget)  #简化后模型显示
        # self.model2_display.setEnabled(True)
        # self.model2_display.setGeometry(QtCore.QRect(500, 480, 400, 400))
        # #self.model2_display.setMinimumSize(QtCore.QSize(400, 400))
        # self.model2_display.setAutoFillBackground(False)
        # self.model2_display.setStyleSheet("background-color: rgb(0, 0, 0);")
        # self.model2_display.setObjectName("model2_display")
         self.label_file_size = QtWidgets.QLabel(self.centralwidget)
         self.label_file_size.setGeometry(QtCore.QRect(300, 150, 200, 40))
         self.label_file_size.setMinimumSize(QtCore.QSize(200, 40))

         self.label_vertex_count = QtWidgets.QLabel(self.centralwidget)
         self.label_vertex_count.setGeometry(QtCore.QRect(300, 250, 200, 40))
         self.label_vertex_count.setMinimumSize(QtCore.QSize(200, 40))

         #控件连接
         self.retranslateUi(simplify)
         self.port_list.currentIndexChanged.connect(self.Mode_Selected)  # 下拉选择
         self.load_model.clicked.connect(self.Load_Model)
         self.text_edit.editingFinished.connect(self.Ratio_Enter)#简化率输入
        # self.load_model.clicked.connect(self.Load_Model)
         QtCore.QMetaObject.connectSlotsByName(simplify)
       #  simplify.setLayout(layout)




     def retranslateUi(self, simplify):
         _translate = QtCore.QCoreApplication.translate
         simplify.setWindowTitle(_translate("simplify", "三维模型轻量化"))
         self.load_model.setText(_translate("simplify","导入模型"))
         print(2)
         self.label_ratio.setText(_translate("simplify", "输入简化率(0-1)："))
         print(1)
 #        self.text_edit.setText(_translate("simplify","输入简化率"))
         self.label_model1.setText(_translate("simplify", "原模型数据"))
         #print(3)
         self.label_model2.setText(_translate("simplify", "简化后模型数据"))
         self.label_model3.setText(_translate("simplify", "文件体积："))
         self.label_model4.setText(_translate("simplify", "顶点数量："))
         self.label_model5.setText(_translate("simplify", "面片数量："))
         self.label_model6.setText(_translate("simplify", "简化后文件体积："))
         self.label_model7.setText(_translate("simplify", "简化后顶点数量："))
         self.label_model8.setText(_translate("simplify", "简化后面片数量："))



         self.label_mode_selected.setText(_translate("simplify","模式选择:"))
         print(5)

