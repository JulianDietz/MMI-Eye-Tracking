import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv("final_data/18_02_20_ba_sum.csv", header=0, delimiter=",", nrows=15)
df = df[['gaze_direction_0_x','gaze_direction_0_y', 'gaze_direction_0_z', 'gaze_direction_1_x', 'gaze_direction_1_y', 'gaze_direction_1_z',
       'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
       'gaze_direction_left_x', 'gaze_direction_left_y', 'gaze_direction_left_z', 'gaze_start_left_x',
       'gaze_start_left_y', 'gaze_start_left_z', 'gaze_start_right_x', 'gaze_start_right_y', 'gaze_start_right_z',
       'gaze_direction_right_x', 'gaze_direction_right_y', 'gaze_direction_right_z', 'aoi_hit_lefteye_Oberer_Küchenschrank_1',
       'aoi_hit_lefteye_Oberer_Küchenschrank_2', 'aoi_hit_lefteye_Oberer_Küchenschrank_3', 'aoi_hit_lefteye_Oberer_Küchenschrank_4',
       'aoi_hit_lefteye_Oberer_Küchenschrank_5', 'aoi_hit_lefteye_Oberer_Küchenschrank_6', 'aoi_hit_lefteye_Oberer_Küchenschrank_7',
       'aoi_hit_lefteye_Unterer_Küchenschrank_1', 'aoi_hit_lefteye_Unterer_Küchenschrank_2', 'aoi_hit_lefteye_Unterer_Küchenschrank_3_Oben',
       'aoi_hit_lefteye_Unterer_Küchenschrank_3_Mitte', 'aoi_hit_lefteye_Unterer_Küchenschrank_3_Unten', 'aoi_hit_lefteye_Unterer_Küchenschrank_4',
       'aoi_hit_lefteye_Unterer_Küchenschrank_5', 'aoi_hit_lefteye_Kühlschrank', 'aoi_hit_lefteye_Kühlschrank-Seitenansicht',
       'aoi_hit_lefteye_Mikrowelle', 'aoi_hit_lefteye_Geschirrspüler', 'aoi_hit_lefteye_Herd_Schalthebel_1','aoi_hit_lefteye_Herd_Schalthebel_2',
       'aoi_hit_lefteye_Herd_Zeitanzeige', 'aoi_hit_lefteye_Herd', 'aoi_hit_lefteye_Cerankochfeld_Oben_Links','aoi_hit_lefteye_Cerankochfeld_Oben_Rechts',
       'aoi_hit_lefteye_Cerankochfeld_Unten_Links', 'aoi_hit_lefteye_Cerankochfeld_Unten_Rechts', 'aoi_hit_lefteye_Spüle', 'aoi_hit_lefteye_Bereich_hinter_Spüle',
       'aoi_hit_lefteye_Bereich_vor_Spüle', 'aoi_hit_lefteye_Arbeitsfläche_1', 'aoi_hit_lefteye_Arbeitsfläche_2', 'aoi_hit_lefteye_Arbeitsfläche_3',
       'aoi_hit_lefteye_Schrankunterseiten', 'aoi_hit_lefteye_Boden', 'aoi_hit_lefteye_Decke', 'aoi_hit_lefteye_Mauer_Links',
       'aoi_hit_lefteye_Mauer_rechts', 'aoi_hit_lefteye_Hinten', 'aoi_hit_lefteye_Arbeitsfläche_1_Vorne', 'aoi_hit_lefteye_Arbeitsfläche_Spüle_Vorne',
       'aoi_hit_lefteye_Arbeitsfläche_2_Vorne', 'aoi_hit_lefteye_Arbeitsfläche_Ofen_Vorne', 'aoi_hit_lefteye_Arbeitsfläche_3_Vorne',
       'aoi_hit_lefteye_Rückwand_ABF_1', 'aoi_hit_lefteye_Rückwand_ABF_Spüle', 'aoi_hit_lefteye_Rückwand_ABF_2',
       'aoi_hit_lefteye_Rückwand_ABF_Ofen', 'aoi_hit_lefteye_Rückwand_ABF_3', 'aoi_hit_righteye_Oberer_Küchenschrank_1',
       'aoi_hit_righteye_Oberer_Küchenschrank_2', 'aoi_hit_righteye_Oberer_Küchenschrank_3', 'aoi_hit_righteye_Oberer_Küchenschrank_4',
       'aoi_hit_righteye_Oberer_Küchenschrank_5', 'aoi_hit_righteye_Oberer_Küchenschrank_6', 'aoi_hit_righteye_Oberer_Küchenschrank_7',
       'aoi_hit_righteye_Unterer_Küchenschrank_1', 'aoi_hit_righteye_Unterer_Küchenschrank_2', 'aoi_hit_righteye_Unterer_Küchenschrank_3_Oben',
       'aoi_hit_righteye_Unterer_Küchenschrank_3_Mitte', 'aoi_hit_righteye_Unterer_Küchenschrank_3_Unten', 'aoi_hit_righteye_Unterer_Küchenschrank_4',
       'aoi_hit_righteye_Unterer_Küchenschrank_5', 'aoi_hit_righteye_Kühlschrank', 'aoi_hit_righteye_Kühlschrank-Seitenansicht',
       'aoi_hit_righteye_Mikrowelle', 'aoi_hit_righteye_Geschirrspüler', 'aoi_hit_righteye_Herd_Schalthebel_1', 'aoi_hit_righteye_Herd_Schalthebel_2',
       'aoi_hit_righteye_Herd_Zeitanzeige', 'aoi_hit_righteye_Herd', 'aoi_hit_righteye_Cerankochfeld_Oben_Links', 'aoi_hit_righteye_Cerankochfeld_Oben_Rechts',
       'aoi_hit_righteye_Cerankochfeld_Unten_Links', 'aoi_hit_righteye_Cerankochfeld_Unten_Rechts', 'aoi_hit_righteye_Spüle',
       'aoi_hit_righteye_Bereich_hinter_Spüle', 'aoi_hit_righteye_Bereich_vor_Spüle', 'aoi_hit_righteye_Arbeitsfläche_1',
       'aoi_hit_righteye_Arbeitsfläche_2', 'aoi_hit_righteye_Arbeitsfläche_3', 'aoi_hit_righteye_Schrankunterseiten', 'aoi_hit_righteye_Boden',
       'aoi_hit_righteye_Decke', 'aoi_hit_righteye_Mauer_Links', 'aoi_hit_righteye_Mauer_rechts', 'aoi_hit_righteye_Hinten',
       'aoi_hit_righteye_Arbeitsfläche_1_Vorne', 'aoi_hit_righteye_Arbeitsfläche_Spüle_Vorne','aoi_hit_righteye_Arbeitsfläche_2_Vorne',
       'aoi_hit_righteye_Arbeitsfläche_Ofen_Vorne', 'aoi_hit_righteye_Arbeitsfläche_3_Vorne', 'aoi_hit_righteye_Rückwand_ABF_1',
       'aoi_hit_righteye_Rückwand_ABF_Spüle', 'aoi_hit_righteye_Rückwand_ABF_2', 'aoi_hit_righteye_Rückwand_ABF_Ofen',
       'aoi_hit_righteye_Rückwand_ABF_3'
]]

df = df.fillna(0).to_numpy().astype(np.float32)
prediction_data = [[df]]

print(prediction_data)
model = tf.keras.models.load_model(
    'models/RUN_15/models/cam_3/activity_recognizer_RUN_15_cam_3_NN.h5',
    custom_objects=None,
    compile=False
)

result = model.predict(prediction_data)
print("Result:")
classes = ["Keine_Klasse", "Konzentriert_arbeiten","Suchen_Oberschrank","Suchen_Unterschrank","Wartend","Keine_Person_erkannt"]
for i, pred_class in enumerate(classes):
    print(pred_class + ': ' + str(round(result[0][i] * 100, 2)))
print(result)
