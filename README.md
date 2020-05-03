# MMI_Eyetracking

## Projektstruktur

* `/action_classification`: Enthält die aufgenommenen und gelabelten Daten, Datenvorverarbeitung, Aktivitätsklassifikatoren und Code zur Aktvitätsbestimmung

* `/classifier`: Enthält den Klassifikator zur Nutzererkennung basierend auf den Beispielbildern

* `/eyetracking`: Enthält den Clienten zur Experimentaufnahme und Visualisierung der AOI hits sowie den Positionserkenner

* `/unity_project_aois_kitchen`: Enthält ein unity Projekt zur Visualisierung verschiedener AOI Konfigurationen

Das Projekt basiert auf vorherigen Arbeiten von Bosek et al. und Guder und enthält deshalb für die Blickbestimmung wichtige Codeabschnitte.


### /action_classification
* /final_data: Für das Training verwendete Daten
* /models: Trainierte Klassifikatoren und Trainingsergebnisse
* keras-classifier.ipynb: Code für das Vorverarbeiten der Daten und Trainieren der Netze 


### /classifier
* /demos/classifier.py: Enthält Code zum Trainieren des Personenerkenners
* /train_images/raw: Enthält Ordner mit Bildern der einzelnen Personen
* /demos/classifier_webcam.py: Führt den Personenerkenner aus und streamt die Ergebnisse über LSL


### /eyetracking
* eyetracking.py: Startet OpenFace und den Clienten
* /config: Enthält die erstellten AOI Konfigurationen

