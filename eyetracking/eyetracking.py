import json
import subprocess
import threading
import ast
from os import path
import pylsl
import tensorflow as tf
import numpy as np
from GazeDetection import GazeDetection
import time
from captureTool import setupCaptureTool, setAOI,setPrediction
from VideoCapture import ViedoCapture


# starts OpenFace with eyetracking and gets the data
def startSingelCam(name, device, callback):
    print('start:' + name + ' ' + device)
    # path of the OpenFace installation
    proc = subprocess.Popen(
        '/Users/juliandietz/Documents/uni/M4/MMI/MMI_Eyetracking/OpenFace/build/bin/FeatureExtraction ' + device,
        shell=True,
        stdout=subprocess.PIPE)
    while True:
        input = proc.stdout.readline()
        input = input.decode('utf-8').rstrip()
        if input.startswith("<relevant_entry>") and input.endswith('</relevant_entry>'):
            message_to_push = str(input)[16:len(input) - 17]
            message_to_push = ast.literal_eval(message_to_push)
            message_to_push['timestamp'] = time.time()
            message_to_push['client_id'] = name
            callback(message_to_push)

# process the raw OpenFace data
def callback(message):
    try:
        # calculates the viewed objects and the positon of the user
        output = gazeDetection.main_method(message)
        setAOI(output['left']['aoi_hits'], output['right']['aoi_hits'])
        stream.push_data(output)
        prediction.calculate_prediction(output, message)
        recorder.record_frame(message, output)
    except Exception as e:
        print(e)


# handles the recording
class RecordData():
    ACTIVITIES_CLASSES = ["Keine_Klasse","Keine_Person_erkannt","Konzentriert_arbeiten",
        "Suchen_Oberschrank","Suchen_Unterschrank","Wartend"]
    participant = None
    record_dir = None
    joined_data = []
    recording = False
    capture_video = False

    def __init__(self):
        self.activity_class = "Keine_Klasse"
        if self.capture_video:
            self.videocapture = ViedoCapture(cam_dict)

    def start_logging(self):
        if self.capture_video:
            self.start_capture_cams()
        self.recording = True
        return self.recording

    def stop_logging(self):
        self.csv_file.close()
        if self.capture_video:
            self.videocapture.stopRecording()
        self.recording = False
        # self.save_csv()

    # callback form the ui-client
    def change_activity_class(self, activity):
        # print('activity changed:' + activity)
        self.activity_class = activity

    '''def save_csv(self):
        with open('config/header_csv_config.json', encoding='utf-8') as f:
            headers = json.load(f)
            joined_header = headers['raw_header_update'] + headers['calculated_header']
        self.write_file(joined_header, self.joined_data)

    def write_file(self, header, data_list):
        try:
            with open(self.filepath + '.csv', 'a') as csv_a, open(self.filepath + '.csv', 'r') as csv_r:
                reader = csv.reader(csv_r)
                writer = csv.DictWriter(csv_a, fieldnames=header)
                file = [row for row in reader]  # turns all the cells into a list
                if len(file) == 0:
                    writer.writeheader()
                for data in data_list:
                    writer.writerow(data)
        except IOError:
            print("Failed to write file")'''

    def open_csv_file(self):
        if not path.exists(self.filepath + '.csv'):
            with open('config/header_csv_config.json', encoding='utf-8') as f:
                headers = json.load(f)
                joined_header = headers['logging_header']
        else:
            joined_header = None

        self.csv_file = open(self.filepath + '.csv', 'a')
        if joined_header:
            for item in joined_header:
                self.csv_file.write("%s," % item)

    def write_row(self, result):
        self.csv_file.write('\n')
        # list=[]
        for key, value in result.items():
            # list.append(key)
            self.csv_file.write("%s," % value)
        # print(list)

    # joins the calculated data to the raw message
    def record_frame(self, message, output):
        if self.recording:
            calculated = {"class": self.activity_class,
                          "gaze_direction_left_x": output['left']['gaze_direction'][0],
                          "gaze_direction_left_y": output['left']['gaze_direction'][1],
                          "gaze_direction_left_z": output['left']['gaze_direction'][2],
                          "gaze_start_left_x": output['left']['gaze_start'][0][0],
                          "gaze_start_left_y": output['left']['gaze_start'][0][1],
                          "gaze_start_left_z": output['left']['gaze_start'][0][2],
                          "gaze_start_right_x": output['right']['gaze_start'][0][0],
                          "gaze_start_right_y": output['right']['gaze_start'][0][1],
                          "gaze_start_right_z": output['right']['gaze_start'][0][2],
                          "gaze_direction_right_x": output['right']['gaze_direction'][0],
                          "gaze_direction_right_y": output['right']['gaze_direction'][1],
                          "gaze_direction_right_z": output['right']['gaze_direction'][2]}
            calculated.update(output['left']['aoi_dict'])
            calculated.update(output['right']['aoi_dict'])
            record_data = {**message, **calculated}
            self.joined_data.append(record_data)
            # stream_raw.push_data(record_data)
            self.write_row({**message, **calculated})

    def start_capture_cams(self):
        cap = threading.Thread(target=self.videocapture.capture)
        cap.start()

    def set_filename(self, directory, participant):
        self.participant = participant
        self.record_dir = directory
        if self.capture_video:
            self.videocapture.init(self.record_dir, self.participant)
        self.filepath = directory + '/' + participant
        self.open_csv_file()
        return self.filepath

# handles the prediction
class PredictActivity():
    paths = {'cam_1': '../action_classification/models/RUN_15/models/cam_1/activity_recognizer_RUN_15_cam_1_NN.h5',
             'cam_2': '../action_classification/models/RUN_15/models/cam_2/activity_recognizer_RUN_15_cam_2_NN.h5',
             'cam_3': '../action_classification/models/RUN_15/models/cam_3/activity_recognizer_RUN_15_cam_3_NN.h5'}
    prediction_size = 15  # (15, 122)
    pred_data = {}
    active = True
    models = {}
    results = {}

    def __init__(self,classes):
        if self.active:
            self.classes=classes
            self.create_Stream()
            self.load_model()

    # creates the Prediction output stream
    def create_Stream(self):
        outlet_info = pylsl.StreamInfo(
            "MMIActivityStream",  # name
            "Activity Prediction data",  # type
            3,  # channel_count
            pylsl.IRREGULAR_RATE,  # samplerate
            pylsl.cf_string,  # channel_format
            "MMIID")  # source_id

        channels = outlet_info.desc().append_child("channels")

        cam1_channel = channels.append_child("channel")
        cam1_channel.append_child_value("label", "cam 1 activity channel")
        cam1_channel.append_child_value("type", "activity prediction")
        cam1_channel.append_child_value("additional_info", "activity prediction")

        cam2_channel = channels.append_child("channel")
        cam2_channel.append_child_value("label", "cam 2 activity channel")
        cam2_channel.append_child_value("type", "activity prediction")
        cam2_channel.append_child_value("additional_info", "activity prediction")

        cam3_channel = channels.append_child("channel")
        cam3_channel.append_child_value("label", "cam 3 activity channel")
        cam3_channel.append_child_value("type", "activity prediction")
        cam3_channel.append_child_value("additional_info", "activity prediction")
        self.outlet = pylsl.StreamOutlet(outlet_info)

    def load_model(self):
        for cam in self.paths:
            model = tf.keras.models.load_model(
                self.paths[cam],
                custom_objects=None,
                compile=False
            )
            self.pred_data[cam] = []
            self.models[cam] = model

    def calculate_prediction(self, output, message):
        if self.active:
            # feeds the data
            cam = message['client_id']
            if len(self.pred_data[cam]) >= self.prediction_size:
                self.pred_data[cam].pop(0)
            new_data = [message['gaze_direction_0_x'], message['gaze_direction_0_y'], message['gaze_direction_0_z'],
                        message['gaze_direction_1_x'], message['gaze_direction_1_y'], message['gaze_direction_1_z'],
                        message['gaze_angle_x'], message['gaze_angle_y'], message['pose_Tx'], message['pose_Ty'],
                        message['pose_Tz'], message['pose_Rx'], message['pose_Ry'], message['pose_Rz'],
                        output['left']['gaze_direction'][0], output['left']['gaze_direction'][1],
                        output['left']['gaze_direction'][2], output['left']['gaze_start'][0][0],
                        output['left']['gaze_start'][0][1], output['left']['gaze_start'][0][2],
                        output['right']['gaze_start'][0][0], output['right']['gaze_start'][0][1],
                        output['right']['gaze_start'][0][2], output['right']['gaze_direction'][0],
                        output['right']['gaze_direction'][1], output['right']['gaze_direction'][2]]
            new_data.extend(output['left']['aoi_dict'].values())
            new_data.extend(output['right']['aoi_dict'].values())
            new_data = np.array(new_data)
            new_data = np.nan_to_num(new_data)
            self.pred_data[cam].append(new_data)

            # calculates the prediction
            for cam in self.pred_data:
                if len(self.pred_data[cam]) == self.prediction_size:
                    prediction_data = np.array([self.pred_data[cam]])
                    result = self.models[cam].predict(prediction_data)
                    print("Result Activity "+cam+":")
                    result_dict = {}
                    for i, pred_class in enumerate(self.classes):
                        result_dict[pred_class] = round(result[0][i] * 100, 2)
                    # result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}
                    print(result_dict)
                    self.results[cam]=result_dict
            self.push_data()
            setPrediction(self.results)

    def push_data(self):
        output=[]
        for cam in self.paths:
            if cam not in self.results:
                output.append(json.dumps({}))
            else:
                output.append(json.dumps(self.results[cam]))
        self.outlet.push_sample(output)


# stream for other components eg. UI or assistance system
class LSLStream():

    def __init__(self):
        self.create_Stream()

    def create_Stream(self):
        outlet_info = pylsl.StreamInfo(
            "MMIEyetrackingStream",  # name
            "Eyetracking data",  # type
            3,  # channel_count
            pylsl.IRREGULAR_RATE,  # samplerate
            pylsl.cf_string,  # channel_format
            "MMIID")  # source_id

        channels = outlet_info.desc().append_child("channels")

        eyeleft_channel = channels.append_child("channel")
        eyeleft_channel.append_child_value("label", "head position")
        eyeleft_channel.append_child_value("type", "head position")
        eyeleft_channel.append_child_value("additional_info", "head position related to camera position")

        eyeleft_channel = channels.append_child("channel")
        eyeleft_channel.append_child_value("label", "left eye channel")
        eyeleft_channel.append_child_value("type", "left eye")
        eyeleft_channel.append_child_value("additional_info", "eyetracking data for the left eye")

        eyeright_channel = channels.append_child("channel")
        eyeright_channel.append_child_value("label", "right eye channel")
        eyeright_channel.append_child_value("type", "right eye")
        eyeright_channel.append_child_value("additional_info", "eyetracking data for the right eye")

        self.outlet = pylsl.StreamOutlet(outlet_info)

    def push_data(self, message):
        self.outlet.push_sample(
            [json.dumps(message['head']), json.dumps(message['left']), json.dumps(message['right'])])


## Stream for logging the raw data
class LSLStreamRaw():

    def __init__(self):
        self.create_Stream()

    def create_Stream(self):
        outlet_info = pylsl.StreamInfo(
            "MMIEyetrackingStreamRaw",  # name
            "Eyetracking data",  # type
            1,  # channel_count
            pylsl.IRREGULAR_RATE,  # samplerate
            pylsl.cf_string,  # channel_format
            "MMIID")  # source_id

        channels = outlet_info.desc().append_child("channels")

        eyeleft_channel = channels.append_child("channel")
        eyeleft_channel.append_child_value("label", "head position")
        eyeleft_channel.append_child_value("type", "head position")
        eyeleft_channel.append_child_value("additional_info", "logging data for the eyetracing classifier")
        self.outlet = pylsl.StreamOutlet(outlet_info)

    def push_data(self, message):
        self.outlet.push_sample([json.dumps(message)])


if __name__ == "__main__":
    # configuration of the cameras; index is the opencv id of the camera and name is camera name of the AOI-Config
    #cam_dict = [{'name': 'cam_3', 'index': 1},{'name': 'cam_2', 'index': 0},{'name': 'cam_1', 'index': 0}]
    cam_dict = [{'name': 'cam_3', 'index': 0}]

    # starts OpenFace Threads
    for cam in cam_dict:
        th = threading.Thread(target=startSingelCam, args=[cam['name'], '-device ' + str(cam['index']), callback])
        th.start()

    gazeDetection = GazeDetection()
    stream = LSLStream()
    # stream_raw = LSLStreamRaw()
    recorder = RecordData()
    prediction = PredictActivity(recorder.ACTIVITIES_CLASSES)
    setupCaptureTool(cam_dict, recorder.ACTIVITIES_CLASSES, recorder.change_activity_class, recorder.start_logging,
                     recorder.stop_logging, recorder.set_filename)