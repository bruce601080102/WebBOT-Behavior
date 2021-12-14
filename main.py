import pymysql
import pandas as pd


def data_from_db(host, db, sql):
    con = pymysql.connect(host=host, user="diia", passwd="diia16313302", db=db, charset='utf8')
    df = pd.read_sql(sql, con)
    return df


def main():
    # sql = "SELECT * FROM robot_info where pk_id > 5018"
    # df_bot = data_from_db(host="192.168.10.201", db="devdb", sql=sql)
    # df_bot.to_pickle("dataset_robot_20211112.pkl")
    # print("df_bot:", len(df_bot))

    sql = "SELECT * FROM robot_info WHERE pk_id >= 275927"
    df_man = data_from_db(host="34.81.27.174", db="diiarelease", sql=sql)
    df_man.to_pickle("dataset_human_20211112.pkl")
    print("df_man:", len(df_man))


if __name__ == '__main__':
    main()
