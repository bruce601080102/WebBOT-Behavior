import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report


def extract_keylogs(logs):
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


def cal_intervaltime(keydown, keyup):
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


def process_keystroke(x):
    logs = json.loads(x["events_log"])
    if len(logs) < 1:
        return [-1]*25

    keydown, keyup, special_key = extract_keylogs(logs)
    if len(keydown) == 0 or len(keyup) == 0:
        return [len(keydown), len(keyup)] + [-1]*23
    if len(keydown) != len(keyup):
        return [len(keydown), len(keyup)] + [-1]*23

    hold, down_down, up_down, double_click = cal_intervaltime(keydown, keyup)
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


def preprocess(df):
    df_event = df[["events_log"]]
    df_event[[
        "len_keydown", "len_keyup", "double_click",
        "mean_hold", "var_hold", "std_hold", "per25_hold", "per50_hold", "per75_hold",
        "mean_dd", "var_dd", "std_dd", "per25_dd", "per50_dd", "per75_dd",
        "mean_ud", "var_ud", "std_ud", "per25_ud", "per50_ud", "per75_ud",
        "backspace", "tab", "shift", "capslock"
    ]] = df_event.apply(process_keystroke, axis=1, result_type="expand")

    df_event = df_event[[
        "mean_hold", "var_hold", "std_hold", "per25_hold", "per50_hold", "per75_hold",
        "mean_dd", "var_dd", "std_dd", "per25_dd", "per50_dd", "per75_dd",
        "mean_ud", "var_ud", "std_ud", "per25_ud", "per50_ud", "per75_ud",
        "backspace", "tab", "shift", "capslock"
    ]]
    return df_event


def validate(clf_dt, scaler, X_test, y_test):
    rescaledX = scaler.transform(X_test)

    y_pred = clf_dt.predict(rescaledX)
    print(classification_report(y_pred, y_test, target_names=['0', '1'], digits=6))

    precision = round(precision_score(y_pred, y_test), 6)
    recall = round(recall_score(y_pred, y_test), 6)
    f1 = round(f1_score(y_pred, y_test), 6)
    accuracy = round(accuracy_score(y_pred, y_test), 6)

    return(precision, recall, f1, accuracy)


def train(df_bot, df_man):
    if len(df_man) > len(df_bot):
        frac = len(df_bot) / len(df_man)
        df_man = df_man.sample(frac=frac).reset_index(drop=True)
    else:
        frac = len(df_man) / len(df_bot)
        df_bot = df_bot.sample(frac=frac).reset_index(drop=True)

    df = pd.concat([df_bot, df_man])

    array = df.values
    X = array[:, :22]
    Y = array[:, 22]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

    clf_dt = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf_dt.fit(rescaledX, y_train)

    # joblib.dump(scaler, 'scaler_keystroke.joblib')
    # joblib.dump(clf_dt, 'model_keystroke.joblib')

    precision, recall, f1, accuracy = validate(clf_dt, scaler, X_test, y_test)
    return precision, recall, f1, accuracy


def main():
    df_bot = pd.read_pickle("./data/dataset/dataset_robot_20211112.pkl")
    df_man = pd.read_pickle("./data/dataset/dataset_human_20211112.pkl")

    df_bot = preprocess(df_bot)
    df_bot["label"] = 1
    print(df_bot)

    df_man = preprocess(df_man)
    df_man = df_man[df_man["capslock"] != -1]
    df_man["label"] = 0
    print(df_man)

    precision, recall, f1, accuracy = train(df_bot, df_man)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("accuracy:", accuracy)


if __name__ == '__main__':
    main()
