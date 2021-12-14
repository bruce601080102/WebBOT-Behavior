# -*- coding: utf-8 -*
import json
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore")

return_code = {
    'pred_success': '4003',                 # 預測成功
    'pred_fail': '9905',                    # 預測失敗
    'train_success': '4007',                # 訓練成功
    'train_fail': '9909',                   # 訓練失敗
    'train_save_fail': '9916',              # 知識庫儲存失敗
    'train_load_fail': '9913',              # 知識庫載入失敗
    'train_vali_fail': '9914',              # 驗證失敗
}


class MouseMove:
    def __init__(self, dataset_path='./', lib_path='./', scaler_mouse=None, model_mouse=None):
        self.dataset_path = dataset_path
        self.lib_path = lib_path
        self.scaler_mouse = scaler_mouse
        self.model_mouse = model_mouse
        self.acc_threshold = 1e-04
        self.counts_threshold = 5

    def _mouse_acceleration(self, info):
        accs, rate_diff = [], []
        same_dff_x, same_dff_y = 0, 0
        zero_acc, mini_acc = 0, 0

        for i in range(len(info)-1):
            delta_t = (info[i+1]["st_t2"] - info[i]["end_t1"]) / 2
            rate_diff.append(abs(info[i+1]["rate"]-info[i]["rate"]))

            acceleration = (info[i+1]["rate"]-info[i]["rate"])/delta_t
            accs.append(acceleration)

            if acceleration == 0:
                zero_acc += 1
            if abs(acceleration) < self.acc_threshold:
                mini_acc += 1
            if info[i]["dff_x"] == info[i+1]["dff_x"]:
                same_dff_x += 1
            if info[i]["dff_y"] == info[i+1]["dff_y"]:
                same_dff_y += 1
        return accs, rate_diff, same_dff_x, same_dff_y, mini_acc, zero_acc

    def _mouse_rate(self, logs):
        def distance(x1, y1, x2, y2):
            return math.pow(math.pow(x2-x1, 2) + math.pow(y2-y1, 2), 1/2)

        rates, counts, info = [], [], []
        horizontal, vertical = 0, 0

        for i in range(len(logs)-1):
            end_t1 = float(logs[i]["timeStamp"])
            end_t2 = float(logs[i+1]["timeStamp"])
            count_t2 = logs[i+1]["count"]
            start_t2 = (end_t2 - (count_t2-1)*100)

            if i < len(logs)-2:
                if logs[i]["pageX"] == logs[i+1]["pageX"] == logs[i+2]["pageX"]:
                    vertical += 1
                if logs[i]["pageY"] == logs[i+1]["pageY"] == logs[i+2]["pageY"]:
                    horizontal += 1

            dist = distance(logs[i]["pageX"], logs[i]["pageY"], logs[i+1]["pageX"], logs[i+1]["pageY"])
            if start_t2-end_t1 == 0:
                rate = dist / 1
            else:
                rate = dist / (start_t2-end_t1)

            rates.append(rate)
            counts.append(logs[i]["count"])
            info.append({
                "rate": rate,
                "end_t1": end_t1,
                "st_t2": start_t2,
                "dff_x": logs[i+1]["pageX"]-logs[i]["pageX"],
                "dff_y": logs[i+1]["pageY"]-logs[i]["pageY"]
            })
        counts.append(logs[-1]["count"])
        return rates, counts, info, horizontal, vertical

    def _mousemove_features(self, logs):
        if len(logs) < 1:
            return [-1]*20

        rates, counts, info, horizontal, vertical = self._mouse_rate(logs)
        if len(rates) == 0:
            return [-1]*20

        accs, rate_diff, same_dff_x, same_dff_y, mini_acc, zero_acc = self._mouse_acceleration(info)
        if len(accs) < 2 or len(rate_diff) < 2:
            return [-1]*20

        per_rates = np.percentile(rates, (25, 50, 75), interpolation='midpoint')
        per_accs = np.percentile(accs, (25, 50, 75), interpolation='midpoint')

        zero_rate_diff = len([x1-x2 for (x1, x2) in zip(rate_diff[:-1], rate_diff[1:]) if x1-x2 == 0])
        stop_counts = len([i for i in counts if i > self.counts_threshold])

        return [
            np.mean(rates), np.var(rates), np.std(rates), per_rates[0], per_rates[1], per_rates[2],
            np.mean(accs), np.var(accs), np.std(accs), per_accs[0], per_accs[1], per_accs[2],
            horizontal, vertical, same_dff_x, same_dff_y, mini_acc, zero_acc, zero_rate_diff, stop_counts
        ]

    def _apply_mousemove(self, x):
        logs = json.loads(x["mouse_log"])
        return self._mousemove_features(logs)

    def _mousemove_preprocess(self, df):
        # df = df[(df["device_type"] == "Desktop or Laptop") | (df["device_type"] == "Virtual Machine")]

        df_mouse = df[["mouse_log"]]
        df_mouse[[
            "mean_rates", "var_rates", "std_rates", "per25_rates", "per50_rates", "per75_rates",
            "mean_accs", "var_accs", "std_accs", "per25_accs", "per50_accs", "per75_accs",
            "horizontal", "vertical", "same_dff_x", "same_dff_y", "mini_acc", "zero_acc", "zero_rate_diff", "stop_counts"
        ]] = df_mouse.apply(self._apply_mousemove, axis=1, result_type="expand")
        df_mouse = df_mouse[df_mouse["zero_rate_diff"] != -1]

        df_mouse = df_mouse[[
            "mean_rates", "var_rates", "std_rates", "per25_rates", "per50_rates", "per75_rates",
            "mean_accs", "var_accs", "std_accs", "per25_accs", "per50_accs", "per75_accs",
            "horizontal", "vertical", "same_dff_x", "same_dff_y", "mini_acc", "zero_acc", "zero_rate_diff", "stop_counts"
        ]]
        return df_mouse

    def _validate_mousemove(self, X_test, y_test):
        rescaledX = self.scaler_mouse.transform(X_test)

        y_pred = self.model_mouse.predict(rescaledX)
        # print(classification_report(y_pred, y_test, target_names=['0', '1'], digits=6))

        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        accuracy = accuracy_score(y_pred, y_test)
        return(precision, recall, f1, accuracy)

    def _report(self, return_code, accuracy=-1, precision=-1, recall=-1, f1=-1):
        return {
            'return_code': return_code,
            'report': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }

    def train_mousemove(self, df_bot, df_man):
        try:
            df_bot = self._mousemove_preprocess(df_bot)
            df_bot["label"] = 1

            df_man = self._mousemove_preprocess(df_man)
            df_man = df_man[
                (df_man["stop_counts"] > 0) & (df_man["zero_rate_diff"] < 2) & (df_man["zero_acc"] < 2) &
                (df_man["horizontal"] < 10) & (df_man["vertical"] < 10)
            ]
            df_man["label"] = 0

            if len(df_man) > len(df_bot):
                frac = len(df_bot) / len(df_man)
                df_man = df_man.sample(frac=frac).reset_index(drop=True)
            else:
                frac = len(df_man) / len(df_bot)
                df_bot = df_bot.sample(frac=frac).reset_index(drop=True)
            df_data = pd.concat([df_bot, df_man])

            array = df_data.values
            X = array[:, :20]
            Y = array[:, 20]

            self.scaler_mouse = StandardScaler().fit(X)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            rescaledX = self.scaler_mouse.transform(X_train)

            self.model_mouse = svm.SVC(kernel='rbf', probability=True)
            self.model_mouse.fit(rescaledX, y_train)
        except Exception as e:
            print("train mousemove fail:", e)
            return self._report(return_code['train_fail'])

        try:
            joblib.dump(self.scaler_mouse, self.lib_path+'scaler_mouse.joblib')
            joblib.dump(self.model_mouse, self.lib_path+'model_mouse.joblib')
        except Exception as e:
            print("train mousemove save fail:", e)
            return self._report(return_code['train_save_fail'])

        try:
            self.scaler_mouse = joblib.load(self.lib_path+'scaler_mouse.joblib')
            self.model_mouse = joblib.load(self.lib_path+'model_mouse.joblib')
        except Exception:
            print("train mousemove load fail")
            return self._report(return_code['train_load_fail'])

        try:
            precision, recall, f1, accuracy = self._validate_mousemove(X_test, y_test)
        except Exception:
            print("train mousemove vali fail")
            return self._report(return_code['train_vali_fail'])

        return self._report(return_code['train_success'], accuracy, precision, recall, f1)

    def predict_mousemove(self, mouse_log=[]):
        if self.scaler_mouse is None or self.model_mouse is None:
            return None

        features = self._mousemove_features(mouse_log)
        if features[-2] == -1:
            return [-1, -1]

        X = np.array([features])

        rescaledX = self.scaler_mouse.transform(X)
        y_prob = self.model_mouse.predict_proba(rescaledX)
        y_prob = y_prob.tolist()
        return(y_prob[0])


class Keystroke:
    def __init__(self, dataset_path='./', lib_path='./', scaler_keystroke=None, model_keystroke=None):
        self.dataset_path = dataset_path
        self.lib_path = lib_path
        self.scaler_keystroke = scaler_keystroke
        self.model_keystroke = model_keystroke

    def _extract_keylogs(self, logs):
        keydown, keyup = [], []
        special_key = {"backspace": 0, "tab": 0, "shift": 0, "capslock": 0}

        for log in logs:
            if log["event"] != "keydown" and log["event"] != "keyup":
                continue
            if "key" not in log:
                return [], [], special_key

            if log["key"] in special_key.keys():
                special_key[log["key"]] += 1

            if log["input_order"] is None:
                continue

            if log["event"] == "keydown":
                keydown.append(log)
            if log["event"] == "keyup":
                keyup.append(log)
        return keydown, keyup, special_key

    def _cal_intervaltime(self, keydown, keyup):
        hold, down_down, up_down = [], [], []
        double_click = 0

        for i in range(len(keydown)):
            hold.append((keyup[i]["time"]-keydown[i]["time"]) / 1000)

            if i > 0 and keydown[i]["input_order"] == keydown[i-1]["input_order"]:
                down_down.append((keydown[i]["time"]-keydown[i-1]["time"]) / 1000)
                up_down.append((keydown[i]["time"]-keyup[i-1]["time"]) / 1000)

                if keydown[i]["time"]-keyup[i-1]["time"] < 0:
                    double_click += 1
        return hold, down_down, up_down, double_click

    def _keystroke_features(self, logs):
        if len(logs) < 1:
            return [-1]*25

        keydown, keyup, special_key = self._extract_keylogs(logs)
        if len(keydown) == 0 or len(keyup) == 0:
            return [len(keydown), len(keyup)] + [-1]*23
        if len(keydown) != len(keyup):
            return [len(keydown), len(keyup)] + [-1]*23

        hold, down_down, up_down, double_click = self._cal_intervaltime(keydown, keyup)
        if double_click > 0:
            return [len(keydown), len(keyup), double_click] + [-1]*22

        if len(hold) == 0 or len(down_down) == 0 or len(up_down) == 0:
            return [len(keydown), len(keyup), double_click] + [-1]*22

        per_hold = np.percentile(hold, (25, 50, 75), interpolation='midpoint')
        per_dd = np.percentile(down_down, (25, 50, 75), interpolation='midpoint')
        per_ud = np.percentile(up_down, (25, 50, 75), interpolation='midpoint')

        return [
            len(keydown), len(keyup), double_click,
            np.mean(hold), np.var(hold), np.std(hold), per_hold[0], per_hold[1], per_hold[2],
            np.mean(down_down), np.var(down_down), np.std(down_down), per_dd[0], per_dd[1], per_dd[2],
            np.mean(up_down), np.var(up_down), np.std(up_down), per_ud[0], per_ud[1], per_ud[2],
            special_key["backspace"], special_key["tab"], special_key["shift"], special_key["capslock"]
        ]

    def _apply_keystroke(self, x):
        logs = json.loads(x["events_log"])
        return self._keystroke_features(logs)

    def _keystroke_preprocess(self, df):
        df_event = df[["events_log"]]
        df_event[[
            "len_keydown", "len_keyup", "double_click",
            "mean_hold", "var_hold", "std_hold", "per25_hold", "per50_hold", "per75_hold",
            "mean_dd", "var_dd", "std_dd", "per25_dd", "per50_dd", "per75_dd",
            "mean_ud", "var_ud", "std_ud", "per25_ud", "per50_ud", "per75_ud",
            "backspace", "tab", "shift", "capslock"
        ]] = df_event.apply(self._apply_keystroke, axis=1, result_type="expand")

        df_event = df_event[[
            "mean_hold", "var_hold", "std_hold", "per25_hold", "per50_hold", "per75_hold",
            "mean_dd", "var_dd", "std_dd", "per25_dd", "per50_dd", "per75_dd",
            "mean_ud", "var_ud", "std_ud", "per25_ud", "per50_ud", "per75_ud",
            "backspace", "tab", "shift", "capslock"
        ]]
        return df_event

    def _validate_keystroke(self, X_test, y_test):
        rescaledX = self.scaler_keystroke.transform(X_test)

        y_pred = self.model_keystroke.predict(rescaledX)
        # print(classification_report(y_pred, y_test, target_names=['0', '1'], digits=6))

        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        accuracy = accuracy_score(y_pred, y_test)

        return(precision, recall, f1, accuracy)

    def _report(self, return_code, accuracy=-1, precision=-1, recall=-1, f1=-1):
        return {
            'return_code': return_code,
            'report': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }

    def train_keystroke(self, df_bot, df_man):
        try:
            df_bot = self._keystroke_preprocess(df_bot)
            df_bot["label"] = 1

            df_man = self._keystroke_preprocess(df_man)
            df_man = df_man[df_man["capslock"] != -1]
            df_man["label"] = 0

            if len(df_man) > len(df_bot):
                frac = len(df_bot) / len(df_man)
                df_man = df_man.sample(frac=frac).reset_index(drop=True)
            else:
                frac = len(df_man) / len(df_bot)
                df_bot = df_bot.sample(frac=frac).reset_index(drop=True)
            df_data = pd.concat([df_bot, df_man])

            array = df_data.values
            X = array[:, :22]
            Y = array[:, 22]

            self.scaler_keystroke = StandardScaler().fit(X)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            rescaledX = self.scaler_keystroke.transform(X_train)

            self.model_keystroke = DecisionTreeClassifier(max_depth=3, random_state=0)
            self.model_keystroke.fit(rescaledX, y_train)
        except Exception as e:
            print("train keystroke fail:", e)
            return self._report(return_code['train_fail'])

        try:
            joblib.dump(self.scaler_keystroke, self.lib_path+'scaler_keystroke.joblib')
            joblib.dump(self.model_keystroke, self.lib_path+'model_keystroke.joblib')
        except Exception as e:
            print("train keystroke save fail:", e)
            return self._report(return_code['train_fail'])

        try:
            self.scaler_keystroke = joblib.load(self.lib_path+'scaler_keystroke.joblib')
            self.model_keystroke = joblib.load(self.lib_path+'model_keystroke.joblib')
        except Exception:
            print("train keystroke load fail")
            return self._report(return_code['train_load_fail'])

        try:
            precision, recall, f1, accuracy = self._validate_keystroke(X_test, y_test)
        except Exception:
            print("train keystroke vali fail")
            return self._report(return_code['train_vali_fail'])

        return self._report(return_code['train_success'], accuracy, precision, recall, f1)

    def predict_keystroke(self, events_log=[]):
        if self.scaler_keystroke is None or self.model_keystroke is None:
            return None

        features = self._keystroke_features(events_log)
        if features[0] != features[1]:
            return [1.0, 0.0]
        if features[2] > 0:
            return [1.0, 0.0]
        if features[-1] == -1:
            return[-1, -1]

        X = np.array([features[3:]])

        rescaledX = self.scaler_keystroke.transform(X)
        y_prob = self.model_keystroke.predict_proba(rescaledX)
        y_prob = y_prob.tolist()
        return(y_prob[0])


class BioBehavior(MouseMove, Keystroke):
    def __init__(self, dataset_path='./', kg_path='./', lib_path='./'):
        MouseMove.__init__(self)
        Keystroke.__init__(self)
        self.dataset_path = dataset_path
        self.kg_path = kg_path
        self.lib_path = lib_path
        self.model_threshold = 0.96

    def load_data(self, robot='simulation_info.csv', human='robot_info.csv'):
        try:
            # df_bot = pd.read_pickle(self.dataset_path+robot)
            # df_man = pd.read_pickle(self.dataset_path+human)
            df_bot = pd.read_csv(self.kg_path+robot)
            df_man = pd.read_csv(self.dataset_path+human)
            return df_bot, df_man
        except Exception:
            return None, None

    def load_models(self, mouse_scaler, mouse_model, key_scaler, key_model):
        try:
            self.scaler_mouse = joblib.load(self.lib_path+mouse_scaler)
            self.model_mouse = joblib.load(self.lib_path+mouse_model)
            self.scaler_keystroke = joblib.load(self.lib_path+key_scaler)
            self.model_keystroke = joblib.load(self.lib_path+key_model)
        except Exception as e:
            print(e)

    def train(self, df_bot, df_man):
        report_mouse = self.train_mousemove(df_bot, df_man)
        report_key = self.train_keystroke(df_bot, df_man)

        if report_mouse["return_code"] != return_code['train_success']:
            return report_mouse
        if report_key["return_code"] != return_code['train_success']:
            return report_key

        accuracy = round(report_mouse["report"]["accuracy"]*0.5 + report_key["report"]["accuracy"]*0.5, 2)
        precision = round(report_mouse["report"]["precision"]*0.5 + report_key["report"]["precision"]*0.5, 2)
        recall = round(report_mouse["report"]["recall"]*0.5 + report_key["report"]["recall"]*0.5, 2)
        f1 = round(report_mouse["report"]["f1_score"]*0.5 + report_key["report"]["f1_score"]*0.5, 2)
        return {
            'return_code': return_code['train_success'],
            'report': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        }

    def predict(self, mouse_log=[], events_log=[]):
        reason = "BioBehavior-"

        mouse_pred = self.predict_mousemove(mouse_log=mouse_log)
        # print("mouse pred:", mouse_pred)
        if mouse_pred is None:
            reason += "NoMousemoveModel"
            return ({
                'return_code': return_code['pred_fail'],
                'score': 0,
                'reason': reason
            })

        if max(mouse_pred) > self.model_threshold:
            reason += "normal" if mouse_pred[0] > mouse_pred[1] else "abnormal"
            return ({
                'return_code': return_code['pred_success'],
                'score': round(mouse_pred[1], 6),
                'reason': reason
            })

        keys_pred = self.predict_keystroke(events_log=events_log)
        # print("keys pred:", keys_pred)
        if keys_pred is None:
            reason += "NoKeystrokeModel"
            return ({
                'return_code': return_code['pred_fail'],
                'score': 0,
                'reason': reason
            })

        if mouse_pred[1] == -1 and keys_pred[1] == -1:
            reason += "normal(NoEnoughLogData)"
            return ({
                'return_code': return_code['pred_success'],
                'score': 0,
                'reason': reason
            })
        elif mouse_pred[1] == -1:
            reason += "normal" if keys_pred[0] > keys_pred[1] else "abnormal"
            return ({
                'return_code': return_code['pred_success'],
                'score': round(keys_pred[1], 6),
                'reason': reason
            })
        elif max(keys_pred) > self.model_threshold:
            reason += "normal" if keys_pred[0] > keys_pred[1] else "abnormal"
            return ({
                'return_code': return_code['pred_success'],
                'score': round(keys_pred[1], 6),
                'reason': reason
            })
        else:
            pred = [
                mouse_pred[0]*0.5 + keys_pred[0]*0.5,
                mouse_pred[1]*0.5 + keys_pred[1]*0.5
            ]
            reason += "normal" if pred[0] > pred[1] else "abnormal"
            return ({
                'return_code': return_code['pred_success'],
                'score': round(pred[1], 6),
                'reason': reason
            })


def main():
    bb = BioBehavior(dataset_path="data/dataset/", lib_path="data/lib/")

    # train
    dataset_robot = "dataset_robot.csv"
    dataset_human = "dataset_human.csv"

    df_bot, df_man = bb.load_data(dataset_robot, dataset_human)
    result = bb.train(df_bot, df_man)
    print(result)

    # predict
    mouse_log = [{'count': 1, 'pageX': 250, 'pageY': 133, 'timeStamp': '3909.300'}, {'count': 1, 'pageX': 379, 'pageY': 434, 'timeStamp': '4010.500'}, {'count': 1, 'pageX': 408, 'pageY': 592, 'timeStamp': '4111.800'}, {'count': 1, 'pageX': 355, 'pageY': 654, 'timeStamp': '4213.300'}, {'count': 1, 'pageX': 213, 'pageY': 647, 'timeStamp': '4314.600'}, {'count': 1, 'pageX': 141, 'pageY': 607, 'timeStamp': '4404.600'}, {'count': 1, 'pageX': 179, 'pageY': 539, 'timeStamp': '4505.900'}, {'count': 1, 'pageX': 188, 'pageY': 510, 'timeStamp': '4595.800'}, {'count': 1, 'pageX': 189, 'pageY': 514, 'timeStamp': '4708.400'}, {'count': 1, 'pageX': 189, 'pageY': 533, 'timeStamp': '4798.400'}, {'count': 1, 'pageX': 188, 'pageY': 534, 'timeStamp': '4910.800'}, {'count': 1, 'pageX': 187, 'pageY': 534, 'timeStamp': '4955.800'}]
    # mouse_log = [{'count': 1, 'pageX': 250, 'pageY': 133, 'timeStamp': '3909.300'}, {'count': 1, 'pageX': 379, 'pageY': 434, 'timeStamp': '4010.500'}]
    events_log = [{'event': 'touchstart', 'time': 1636820484178, 'name': 'eeep_cc_cc_number', 'input_order': 'inputOrder0', 'timeStamp': 5519}, {'event': 'mouseenter', 'time': 1636820484269, 'pageX': 292, 'pageY': 319, 'name': 'eeep_cc_cc_number', 'input_order': 'inputOrder0', 'timeStamp': 5609}, {'event': 'click', 'time': 1636820484314, 'pageX': 292, 'pageY': 319, 'name': 'eeep_cc_cc_number', 'input_order': 'inputOrder0', 'timeStamp': 5666}, {'event': 'keydown', 'key': '', 'time': 1636820525263, 'name': 'eeep_cc_cc_number', 'input_order': 'inputOrder0', 'timeStamp': 46642}, {'event': 'keyup', 'key': '', 'time': 1636820525268, 'name': 'eeep_cc_cc_number', 'input_order': 'inputOrder0', 'timeStamp': 46648}, {'event': 'mouseenter', 'time': 1636820537601, 'pageX': 163, 'pageY': 426, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 58985}, {'event': 'click', 'time': 1636820537622, 'pageX': 163, 'pageY': 426, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 59008}, {'event': 'touchstart', 'time': 1636820695342, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 216658}, {'event': 'mouseenter', 'time': 1636820695752, 'pageX': 176, 'pageY': 425, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 217130}, {'event': 'click', 'time': 1636820695773, 'pageX': 176, 'pageY': 425, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 217165}, {'event': 'touchstart', 'time': 1636820752558, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 273947}, {'event': 'mouseenter', 'time': 1636820753000, 'pageX': 173, 'pageY': 414, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 274375}, {'event': 'click', 'time': 1636820753024, 'pageX': 173, 'pageY': 414, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 274416}, {'event': 'keydown', 'key': '', 'time': 1636820754509, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814141240729300}, {'event': 'keyup', 'key': '', 'time': 1636820754533, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814141240729500}, {'event': 'keydown', 'key': 'backspace', 'time': 1636820754765, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814410413062400}, {'event': 'keyup', 'key': 'backspace', 'time': 1636820754787, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814410413062500}, {'event': 'keydown', 'key': '', 'time': 1636820755157, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814802422396500}, {'event': 'keyup', 'key': '', 'time': 1636820755176, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636814802422396500}, {'event': 'keydown', 'key': '', 'time': 1636820755381, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636815026724312300}, {'event': 'keyup', 'key': '', 'time': 1636820755397, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636815026724312300}, {'event': 'keydown', 'key': '', 'time': 1636820755532, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636815177718646500}, {'event': 'keyup', 'key': '', 'time': 1636820755546, 'name': 'CVV2', 'input_order': 'inputOrder1', 'timeStamp': 636815177718646500}, {'event': 'touchstart', 'time': 1636820757879, 'name': '確定付款', 'input_order': None, 'timeStamp': 279261}, {'event': 'mouseenter', 'time': 1636820758300, 'pageX': 215, 'pageY': 820, 'name': 'place-order-button', 'input_order': 'inputOrder2', 'timeStamp': 279691}]
    mouse_scaler = "scaler_mouse.joblib"
    mouse_model = "model_mouse.joblib"
    key_scaler = "scaler_keystroke.joblib"
    key_model = "model_keystroke.joblib"

    bb.load_models(mouse_scaler, mouse_model, key_scaler, key_model)
    result = bb.predict(mouse_log, events_log)
    print(result)


if __name__ == '__main__':
    main()
