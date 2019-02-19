from PyQt5.QtWidgets import *

app = QApplication([])
app.setStyle('Fusion')
layout = QVBoxLayout()
layout.addWidget(QPushButton('Top'))
layout.addWidget(QPushButton('Bottom'))

button = QPushButton('Click')
def on_button_clicked():
	alert = QMessageBox()
	alert.setText('You clicked this')
	alert.exec_()
button.clicked.connect(on_button_clicked)
layout.addWidget(button)

window = QWidget()
window.setLayout(layout)
window.show()
app.exec_()
