import sys
from PyQt5.QtWidgets import *
from inventory.ui import *
from inventory.lib import heuristic


def update():
    kisi_sayisi = ui.input_kisisayisi.text()
    sure = ui.input_sure.text()
    minPartiSayisi = ui.input_minpartisayisi.text()
    maxPartiSayisi = ui.input_maxPartiSayisi.text()
    parti_suresi = ui.input_PartiSuresi.text()
    toplam_sure = ui.input_toplamSure.text()

    """
    materyale_gore=ui.input_materyalegore.text()    

    if ui.zamanagore.isChecked():
        _zamanagore_ = "checked"
    else:
        _zamanagore_ = "not_checked"
    if ui.materyalegore.isChecked():
        _materyalegore_ = "işaretli"
    else:
        _materyalegore_ = "işaretli değil"

        """
    # daily_input_table = f"INSERT INT daily_input_table \
    # (Kişi Sayısı,Süre,Parti Sayısı min, Parti Sayısı max, Zaman Göre, Materyale Göre) \
    # VALUES {kisi_sayisi},{sure},{minPartiSayisi},{maxPartiSayisi},{zamana_gore},{materyale_gore}
    # curs.execute(daily_input_table)
    # conn.commit()

    print(kisi_sayisi)
    print(maxPartiSayisi)
    print(minPartiSayisi)
    print(parti_suresi)
    print(toplam_sure)
    print(sure)
    heuristic(min_batch=int(minPartiSayisi), max_batch=int(maxPartiSayisi), max_batch_time=float(parti_suresi),
              max_total_time=float(toplam_sure), running_time=float(sure), n_people=int(kisi_sayisi), max_wo_count=40)


app = QApplication(sys.argv)
penAna = QMainWindow()
ui = Ui_aselsan_ui()
ui.setupUi(penAna)
penAna.show()
ui.calistir.clicked.connect(update)
sys.exit(app.exec_())
