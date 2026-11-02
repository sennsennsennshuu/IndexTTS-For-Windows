"""深色平面主题：用层级、留白和少量蓝灰色区分操作，不依赖系统调色板。"""

COLORS = {
    "background": "#101215", "surface": "#191c21", "inset": "#13161a",
    "border": "#2b3038", "text": "#e5e8ed", "muted": "#929ba9",
    "quiet": "#626d7c", "accent": "#9bacbf", "accent_dark": "#74879e",
}

STYLESHEET = """
QDialog#studio { background: #101215; }
QWidget { color: #e5e8ed; font-family: 'Microsoft YaHei UI', 'Microsoft YaHei'; font-size: 12px; }
QLabel { background: transparent; border: none; }
QLabel#brand { font-size: 22px; font-weight: 700; letter-spacing: -0.7px; }
QLabel#brandTag { color: #7f8c9d; font-size: 10px; letter-spacing: 2px; }
QLabel#subtitle { color: #87919e; font-size: 12px; }
QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
QLabel#eyebrow { color: #758295; font-size: 10px; letter-spacing: 1.7px; }
QLabel#muted { color: #929ba9; font-size: 11px; }
QLabel#quiet { color: #75808f; font-size: 11px; }
QLabel#status { color: #a9b8ca; font-size: 12px; }
QLabel#status[state='error'] { color: #d1a4a4; }
QLabel#status[state='done'] { color: #b8c9d9; }
QLabel#pill { background: #1b2129; color: #9eafc4; border: 1px solid #303b49; border-radius: 12px; padding: 5px 12px; font-size: 11px; }
QLabel#fileTitle { color: #d4dce7; font-size: 12px; font-weight: 500; }
QLabel#timecode { color: #94a2b4; font-family: 'Consolas'; font-size: 11px; }
QFrame#card { background: #191c21; border: 1px solid #2b3038; border-radius: 12px; }
QFrame#divider { background: #282d35; border: none; max-height: 1px; }
QFrame#footer { background: #101215; border-top: 1px solid #292e36; }
QFrame#audioDrop { background: #15191e; border: 1px dashed #3a4554; border-radius: 9px; }
QFrame#audioDrop[dragging='true'] { background: #202a36; border: 1px solid #9bacbf; }
QFrame#audioDrop[loaded='true'] { border-style: solid; border-color: #384353; }
QTextEdit { background: #191c21; border: none; padding: 2px; color: #e2e6ed; selection-background-color: #3d4d63; selection-color: #ffffff; font-size: 16px; }
QLineEdit { background: #13161a; border: 1px solid #333a45; border-radius: 6px; padding: 9px 10px; color: #d6dde7; selection-background-color: #3d4d63; }
QLineEdit:focus { border: 1px solid #788ca8; }
QPushButton { background: #252b34; color: #c5cfdd; border: 1px solid #394351; border-radius: 6px; padding: 9px 14px; font-size: 12px; }
QPushButton:hover { background: #303947; border-color: #56667d; }
QPushButton:pressed { background: #1e2530; }
QPushButton:disabled { background: #20242a; color: #626d7c; border-color: #2d333d; }
QPushButton#ghost { background: transparent; border: 1px solid transparent; color: #9ca8ba; padding: 6px 8px; }
QPushButton#ghost:hover { background: #262d37; color: #e2e7ee; }
QPushButton#primary { background: #a7b7cb; color: #111820; border: none; font-size: 14px; font-weight: 600; padding: 13px 26px; border-radius: 7px; }
QPushButton#primary:hover { background: #bccadb; }
QPushButton#primary:pressed { background: #8e9fb6; }
QPushButton#primary:disabled { background: #343f4e; color: #8c9bb0; }
QPushButton#play { background: #a7b7cb; color: #111820; border: none; border-radius: 20px; padding: 0; }
QPushButton#play:disabled { background: #303844; }
QPushButton#fold { background: transparent; border: none; color: #a6b2c3; text-align: left; padding: 0; font-size: 12px; }
QPushButton#fold:hover { color: #e5e8ed; }
QPushButton[mode='true'] { background: #20252c; border: 1px solid #303844; color: #8e9aab; padding: 9px 4px; font-size: 11px; }
QPushButton[mode='true']:checked { background: #2b3543; border-color: #667b97; color: #dce5f0; }
QComboBox { background: #13161a; color: #d8e0eb; border: 1px solid #343e4b; border-radius: 6px; padding: 8px; }
QComboBox QAbstractItemView { background: #202630; color: #d8e0eb; selection-background-color: #37475b; outline: none; }
QSpinBox, QDoubleSpinBox { background: #13161a; color: #d5deea; border: 1px solid #333d4a; border-radius: 6px; padding: 7px 10px; min-height: 20px; font-family: 'Consolas'; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7c90ac; }
QSlider { background: transparent; min-height: 18px; max-height: 18px; }
QSlider::groove:horizontal { height: 3px; background: #333c48; border: none; border-radius: 1px; }
QSlider::sub-page:horizontal { height: 3px; background: #8b9fb9; border: none; border-radius: 1px; }
QSlider::add-page:horizontal { height: 3px; background: #333c48; border: none; border-radius: 1px; }
QSlider::handle:horizontal { background: #bac8d9; border: 2px solid #191c21; width: 10px; height: 10px; margin: -6px 0; border-radius: 7px; }
QSlider:disabled::sub-page:horizontal { background: #465364; }
QCheckBox { color: #929eaf; spacing: 7px; font-size: 11px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #526075; border-radius: 3px; background: #15191e; }
QCheckBox::indicator:checked { background: #9bacbf; border: 3px solid #53657b; }
QProgressBar { background: #252e39; border: none; border-radius: 1px; max-height: 2px; min-height: 2px; }
QProgressBar::chunk { background: #9bacbf; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 2px 0; }
QScrollBar::handle:vertical { background: #3b4655; border-radius: 3px; min-height: 35px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #28313f; color: #e2e8f0; border: 1px solid #526176; padding: 6px; }
QMessageBox { background: #191c21; }
"""
