import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog
from PySide6.QtGui import QPixmap
from FormatDesign import Ui_MainWindow
from PIL import Image
from pathlib import Path
import os

#For correct application working use || pip install Pillow, pathlib, PySide6


class MainWindow(QMainWindow):

    # Init function, contains local variables for application

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.scene = QGraphicsScene()
        self.ui.ShowImageView.setScene(self.scene)
        self.file_path = None
        self.picked_format = None
        self.last_format = None
        self.pil_image = None
        self.file_size_w = None
        self.file_size_h = None
        self.precess_state = None

        self.ui.OpenFileButton.clicked.connect(self.open_image)
        self.ui.SaveButton.clicked.connect(self.ConvertSaveFile)
        self.ui.JPEG.toggled.connect(lambda: self.DefineFormat('JPEG'))
        self.ui.SetSize.clicked.connect(self.SetSize)

        #Buttons to Functions connections

        self.ui.PNG.toggled.connect(lambda: self.DefineFormat('PNG'))
        self.ui.BMP.toggled.connect(lambda: self.DefineFormat('BMP'))
        self.ui.TIFF.toggled.connect(lambda: self.DefineFormat('TIFF'))
        self.ui.WebP.toggled.connect(lambda: self.DefineFormat('WebP'))
        self.ui.ICO.toggled.connect(lambda: self.DefineFormat('ICO'))
        self.ui.PPM.toggled.connect(lambda: self.DefineFormat('PPM'))
        self.ui.TGA.toggled.connect(lambda: self.DefineFormat('TGA'))
        self.ui.EPS.toggled.connect(lambda: self.DefineFormat('EPS'))
        self.ui.PDF.toggled.connect(lambda: self.DefineFormat('PDF'))
        self.ui.DDS.toggled.connect(lambda: self.DefineFormat('DDS'))
        self.ui.PCX.toggled.connect(lambda: self.DefineFormat('PCX'))
        self.ui.SGI.toggled.connect(lambda: self.DefineFormat('SGI'))

    #Default open image button for opening file

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "Downloads", "Изображения (*.png *.jpg *.bmp *.jpeg *.tiff *.WebP *.ico *.tga *.psd *.eps *.pdf *.dds *.pcx *.sgi)"
        )

        if not path:
            return
        if path:
            self.file_path = path
            print(self.file_path)
            self.pil_image = Image.open(f"{self.file_path}")

        pixmap = QPixmap(path)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    #General format function, changes picked format variable for save button attributes

    def DefineFormat(self, format_code):
        if format_code == 'JPEG':
            self.picked_format = 'JPEG'
        elif format_code == 'PNG':
            self.picked_format = 'PNG'
        elif format_code == 'TIFF':
            self.picked_format = 'TIFF'
        elif format_code == 'Webp':
            self.picked_format = 'WebP'
        elif format_code == 'ICO':
            self.picked_format = 'ICO'
        elif format_code == 'PPM':
            self.picked_format = 'PPM'
        elif format_code == 'TGA':
            self.picked_format = 'TGA'
        elif format_code == 'PSD':
            self.picked_format = 'PSD'
        elif format_code == 'EPS':
            self.picked_format = 'EPS'
        elif format_code == 'PDF':
            self.picked_format = 'PDF'
        elif format_code == 'DDS':
            self.picked_format = 'DDS'
        elif format_code == 'PCX':
            self.picked_format = 'PCX'
        elif format_code == 'SGI':
            self.picked_format = 'SGI'

        self.ui.FormatLabel.setText(f"{self.picked_format}")
        self.ConState()

        self.last_format = self.picked_format

    #Log info for debug

    def ConState(self):
        if self.last_format != self.picked_format:
            print(f"Convertion State: {self.picked_format}")

    #Save button function

    def ConvertSaveFile(self):
        try:
            format = self.picked_format
            new_path_file = Path(self.file_path)
            new_filepath = new_path_file.with_suffix(f"{"." + self.picked_format}")
            img = Image.open(f"{self.file_path}")
            img = img.convert("RGB")
            new_size = int(self.file_size_w), int(self.file_size_h)
            img = img.resize(new_size)
            img.save(f"{new_filepath}", f"{format}")
            print(f"Saved at {new_filepath} with format {format}")
            self.process_state = "positive"
            self.ui.WARNING.setText(f"Conversion Done at {new_filepath}")
            self.ui.WARNING.setStyleSheet(u"color: green")
        except:
            self.process_state = "negative"
            if self.process_state == "negative":
                self.ui.WARNING.setText("Conversion Failed")

    #Size button function

    def SetSize(self):
        self.ui.SizeLabel.setText(f"{self.ui.width.text()}x{self.ui.height.text()}")
        self.file_size_w, self.file_size_h = self.ui.width.text(), self.ui.height.text()
        if int(self.file_size_h) + int(self.file_size_w) < (2):
            self.ui.WARNING.setText("Incorrect resolution")
        elif int(self.file_size_h) + int(self.file_size_w) >= 4096:
            self.ui.WARNING.setText("WARNING: Double check the resolution parameters")
            self.ui.WARNING.setStyleSheet(u"color: orange")
        else: self.ui.WARNING.setText("")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
