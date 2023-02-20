# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:49:43 2022

@author: m.debritovanvelze
"""


from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

class main_UI(QDialog):
    def __init__(self, parent=None):
        super(main_UI, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        self.create_dataset = QPushButton("&Create a dataset")
        self.create_dataset.setDefault(True)
        self.create_dataset.clicked.connect(self.get_results)
        self.combine_dataset = QPushButton("&Combine datasets")
        self.combine_dataset.setDefault(True)
        self.combine_dataset.clicked.connect(self.get_results)
        self.run_analysis = QPushButton("&Run Analysis")
        self.run_analysis.setDefault(True)
        self.run_analysis.clicked.connect(self.get_results)
        self.plot_combined = QPushButton("&Plot Combined Figure")
        self.plot_combined.setDefault(True)
        self.plot_combined.clicked.connect(self.get_results)
        self.event_analysis = QPushButton("&Run event-based Analysis")
        self.event_analysis.setDefault(True)
        self.event_analysis.clicked.connect(self.get_results)
        
        layout = QGridLayout()
        layout.addWidget(self.create_dataset, 0, 0)
        layout.addWidget(self.combine_dataset, 1, 0)
        layout.addWidget(self.run_analysis, 2, 0)
        layout.addWidget(self.plot_combined, 3, 0)
        layout.addWidget(self.event_analysis, 4, 0)
        self.setLayout(layout)
        self.setWindowTitle("Calcium Analysis Package")
        
        super(main_UI, self).accept()
    
        
if __name__ == '__main__':
    ui = main_UI()
    ui.show()