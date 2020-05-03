import pandas as pd
from pathlib import Path
import statistics

PATH_TO_DATA_FOLDER = "../action_classification/final_data"
PATH_TO_TEST_DATA = "../action_classification/final_data_test/example_kochen.csv"

COLUMN_NAMES = ["client_id", "frame_number", "timestamp"]

frame_rates_cam1 = []
frame_rates_cam2 = []
frame_rates_cam3 = []

path_dir = Path(PATH_TO_DATA_FOLDER)
file_list = [f for f in path_dir.glob('*.csv') if f.is_file()]
#print(file_list)
files_loaded = []
for index, file in enumerate(file_list):
    df = pd.read_csv(str(file))
    try:
           df = df[COLUMN_NAMES]
           df = df.dropna()

           df_cam1 = df[df['client_id'] == "cam_1"]
           df_cam2 = df[df['client_id'] == "cam_2"]
           df_cam3 = df[df['client_id'] == "cam_3"]

           nrow_cam1 = df_cam1.shape[0]
           nrow_cam2 = df_cam2.shape[0]
           nrow_cam3 = df_cam3.shape[0]

           idx_0_cam1 = df_cam1['timestamp'].iloc[0]
           idx_0_cam2 = df_cam2['timestamp'].iloc[0]
           idx_0_cam3 = df_cam3['timestamp'].iloc[0]

           idx_e_cam1 = df_cam1['timestamp'].iloc[-1]
           idx_e_cam2 = df_cam2['timestamp'].iloc[-1]
           idx_e_cam3 = df_cam3['timestamp'].iloc[-1]

           dur_cam1 = idx_e_cam1 - idx_0_cam1
           dur_cam2 = idx_e_cam2 - idx_0_cam2
           dur_cam3 = idx_e_cam3 - idx_0_cam3

           frame_rates_cam1.append(nrow_cam1 / dur_cam1)
           frame_rates_cam2.append(nrow_cam2 / dur_cam2)
           frame_rates_cam3.append(nrow_cam3 / dur_cam3)
    except:
        print("error occurred")

print(statistics.mean(frame_rates_cam1))
print(statistics.mean(frame_rates_cam2))
print(statistics.mean(frame_rates_cam3))