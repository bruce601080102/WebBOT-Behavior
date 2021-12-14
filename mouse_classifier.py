import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report


acc_threshold = 1e-04
counts_threshold = 5


def mouse_acceleration(info):
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
        if abs(acceleration) < acc_threshold:
            mini_acc += 1
        if info[i]["dff_x"] == info[i+1]["dff_x"]:
            same_dff_x += 1
        if info[i]["dff_y"] == info[i+1]["dff_y"]:
            same_dff_y += 1

    return accs, rate_diff, same_dff_x, same_dff_y, mini_acc, zero_acc


def mouse_rate(logs):
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


def process_mouse_move(x):
    logs = json.loads(x["mouse_log"])
    if len(logs) < 1:
        return [-1]*20

    rates, counts, info, horizontal, vertical = mouse_rate(logs)
    if len(rates) == 0:
        return [-1]*20

    accs, rate_diff, same_dff_x, same_dff_y, mini_acc, zero_acc = mouse_acceleration(info)
    if len(accs) < 2 or len(rate_diff) < 2:
        return [-1]*20

    per_rates = np.percentile(rates, (25, 50, 75), interpolation='midpoint')
    per_accs = np.percentile(accs, (25, 50, 75), interpolation='midpoint')

    zero_rate_diff = len([x1-x2 for (x1, x2) in zip(rate_diff[:-1], rate_diff[1:]) if x1-x2 == 0])
    stop_counts = len([i for i in counts if i > counts_threshold])

    return [
        np.mean(rates), np.var(rates), np.std(rates), per_rates[0], per_rates[1], per_rates[2],
        np.mean(accs), np.var(accs), np.std(accs), per_accs[0], per_accs[1], per_accs[2],
        horizontal, vertical, same_dff_x, same_dff_y, mini_acc, zero_acc, zero_rate_diff, stop_counts
    ]


def preprocess(df):
    # df = df[(df["device_type"] == "Desktop or Laptop") | (df["device_type"] == "Virtual Machine")]

    df_mouse = df[["mouse_log"]]
    df_mouse[[
        "mean_rates", "var_rates", "std_rates", "per25_rates", "per50_rates", "per75_rates",
        "mean_accs", "var_accs", "std_accs", "per25_accs", "per50_accs", "per75_accs",
        "horizontal", "vertical", "same_dff_x", "same_dff_y", "mini_acc", "zero_acc", "zero_rate_diff", "stop_counts"
    ]] = df_mouse.apply(process_mouse_move, axis=1, result_type="expand")
    df_mouse = df_mouse[df_mouse["zero_rate_diff"] != -1]

    df_mouse = df_mouse[[
        "mean_rates", "var_rates", "std_rates", "per25_rates", "per50_rates", "per75_rates",
        "mean_accs", "var_accs", "std_accs", "per25_accs", "per50_accs", "per75_accs",
        "horizontal", "vertical", "same_dff_x", "same_dff_y", "mini_acc", "zero_acc", "zero_rate_diff", "stop_counts"
    ]]
    return df_mouse


def validate(clf_rbf, scaler, X_test, y_test):
    rescaledX = scaler.transform(X_test)

    y_pred = clf_rbf.predict(rescaledX)
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
    X = array[:, :20]
    Y = array[:, 20]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

    clf_rbf = svm.SVC(kernel='rbf', probability=True)
    clf_rbf.fit(rescaledX, y_train)

    # joblib.dump(scaler, 'scaler_mouse.joblib')
    # joblib.dump(clf_rbf, 'model_mouse.joblib')

    precision, recall, f1, accuracy = validate(clf_rbf, scaler, X_test, y_test)
    return precision, recall, f1, accuracy


def main():
    df_bot = pd.read_pickle("./data/dataset/dataset_robot_20211112.pkl")
    df_man = pd.read_pickle("./data/dataset/dataset_human_20211112.pkl")

    df_bot = preprocess(df_bot)
    df_bot["label"] = 1
    print(df_bot)

    df_man = preprocess(df_man)
    df_man = df_man[
        (df_man["stop_counts"] > 0) & (df_man["zero_rate_diff"] < 2) & (df_man["zero_acc"] < 2) &
        (df_man["horizontal"] < 10) & (df_man["vertical"] < 10)
    ]
    df_man["label"] = 0
    print(df_man)

    precision, recall, f1, accuracy = train(df_bot, df_man)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("accuracy:", accuracy)


if __name__ == '__main__':
    main()
