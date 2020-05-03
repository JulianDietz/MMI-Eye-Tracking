import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import logging


RELEVANT_VALUES = ['face_id', 'frame_number', 'landmark_detection_success', 'landmark_detection_confidence', 'gaze_direction_0_x',
                   'gaze_direction_0_y', 'gaze_direction_0_z', 'gaze_direction_1_x', 'gaze_direction_1_y', 'gaze_direction_1_z',
                   'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'client_id',
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
                   'aoi_hit_righteye_Rückwand_ABF_3']

CLASSES = {"Keine_Klasse": 0, "Konzentriert_arbeiten": 1, "Suchen_Oberschrank": 2, "Suchen_Unterschrank": 3,
           "Suchen_Arbeitsfläche": 4, "Wartend": 5, "Keine_Person_erkannt": 6}


### read data ###

df = pd.read_csv('../data/example_kochen.csv')
df = df.fillna(0)

data_cam1 = df.loc[df['client_id'] == "cam_1"]
data_cam2 = df.loc[df['client_id'] == "cam_2"]
data_cam3 = df.loc[df['client_id'] == "cam_3"]

data_relevant_cam1 = data_cam1[RELEVANT_VALUES]
data_relevant_cam2 = data_cam2[RELEVANT_VALUES]
data_relevant_cam3 = data_cam3[RELEVANT_VALUES]

labels_cam1 = data_cam1['class']
labels_cam2 = data_cam2['class']
labels_cam3 = data_cam3['class']

### prep labels and test/train sets ###

lb = LabelBinarizer()
lb.fit(df['class'])
labels_cam1 = lb.transform(labels_cam1)
labels_cam2 = lb.transform(labels_cam2)
labels_cam3 = lb.transform(labels_cam3)

num_train_data_cam1 = int(labels_cam1.shape[0] * .8)
num_train_data_cam2 = int(labels_cam2.shape[0] * .8)
num_train_data_cam3 = int(labels_cam3.shape[0] * .8)

num_test_data_cam1 = labels_cam1.shape[0] - num_train_data_cam1
num_test_data_cam2 = labels_cam2.shape[0] - num_train_data_cam2
num_test_data_cam3 = labels_cam3.shape[0] - num_train_data_cam3

# cam 1
x_train_cam1 = data_relevant_cam1[:num_train_data_cam1]
x_test_cam1 = data_relevant_cam1[num_train_data_cam1:]
# cam 2
x_train_cam2 = data_relevant_cam2[:num_train_data_cam2]
x_test_cam2 = data_relevant_cam2[num_train_data_cam2:]
# cam 3
x_train_cam3 = data_relevant_cam3[:num_train_data_cam3]
x_test_cam3 = data_relevant_cam3[num_train_data_cam3:]

# cam 1
y_train_cam1 = labels_cam1[:num_train_data_cam1]
y_test_cam1 = labels_cam1[num_train_data_cam1:]
# cam 2
y_train_cam2 = labels_cam2[:num_train_data_cam2]
y_test_cam2 = labels_cam2[num_train_data_cam2:]
# cam3
y_train_cam3 = labels_cam3[:num_train_data_cam3]
y_test_cam3 = labels_cam3[num_train_data_cam3:]

num_x_signals = data_relevant_cam1.shape[1]
num_y_signals = labels_cam1.shape[1]

def batch_generator_train(batch_size, sequence_length, cam_id):
    if cam_id == 1:
        num_train_data = num_train_data_cam1
        x_train = x_train_cam1
        y_train = y_train_cam1
    elif cam_id == 2:
        num_train_data = num_train_data_cam2
        x_train = x_train_cam2
        y_train = y_train_cam2
    elif cam_id == 3:
        num_train_data = num_train_data_cam3
        x_train = x_train_cam3
        y_train = y_train_cam3
    else:
        logging.info('invalid input')
        return -1

    # Keep on producing batches
    while True:

        # make Placeholders for the data
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)

        y_shape = (batch_size, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Generate a single batch
        batches_added = 0
        while batches_added < batch_size:
            idx = np.random.randint(num_train_data - sequence_length)

            for incrementor in range(0, sequence_length):
                if y_train[idx + incrementor] != y_train[idx]:
                    break

            x_batch[batches_added] = x_train[idx:idx + sequence_length]
            y_batch[batches_added] = y_train[idx + sequence_length - 1]
            batches_added = batches_added + 1

        yield (x_batch, y_batch)


def batch_generator_val(batch_size, sequence_length, cam_id):
    if cam_id == 1:
        num_test_data = num_test_data_cam1
        x_test = x_test_cam1
        y_test = y_test_cam1
    elif cam_id == 2:
        num_test_data = num_test_data_cam2
        x_test = x_test_cam2
        y_test = y_test_cam2
    elif cam_id == 3:
        num_test_data = num_test_data_cam3
        x_test = x_test_cam3
        y_test = y_test_cam3
    else:
        logging.info('invalid input')
        return -1

    # Keep on producing batches
    while True:

        # make Placeholders for the data
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)

        y_shape = (batch_size, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Generate a single batch
        batches_added = 0
        while batches_added < batch_size:
            idx = np.random.randint(num_test_data - sequence_length)

            for incrementor in range(0, sequence_length):
                if y_test[idx + incrementor] != y_test[idx]:
                    break

            x_batch[batches_added] = x_test[idx:idx + sequence_length]
            y_batch[batches_added] = y_test[idx + sequence_length - 1]
            batches_added = batches_added + 1

        yield (x_batch, y_batch)


# cam1
generator_cam1 = batch_generator_train(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=1)
generator_val_cam1 = batch_generator_val(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=1)

# cam2
generator_cam2 = batch_generator_train(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=2)
generator_val_cam2 = batch_generator_val(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=2)

# cam3
generator_cam3 = batch_generator_train(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=3)
generator_val_cam3 = batch_generator_val(batch_size=config["ml_batch_size"], sequence_length=config["ml_sequence_length"], cam_id=3)
