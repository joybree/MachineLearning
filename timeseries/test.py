# -*- coding: utf-8 -*-
import unittest
import arima
import os
import pandas as pd


class Arima_Test(unittest.TestCase):
    def set_data_dir(self):
        print("set_data_dir")
        self.dir = "E:/code/python/MachineLearning/data/test_data/"
        self.error = 0.001
        self.num_percent = 0.9
    
    def test_result_one_point(self):
        true_num = 0
        false_num = 0

        print("****test_result_compare****")
        self.set_data_dir()
        filelist = os.listdir(self.dir)
        list_ts_data = []
        for file_name in filelist:
            df_data = pd.read_csv(self.dir+file_name, encoding='utf-8', index_col='date')
            df_data.index = pd.to_datetime(df_data.index)
            ts_data = df_data['value']
            list_ts_data.append(ts_data)
            prediction_value, prediction_var, prediction_con = arima.prediction(ts_data, pre_num=1)
            print(prediction_value[0])
            print(ts_data[-1])
            if abs(prediction_value[0] - ts_data[-1])/ts_data[-1] <= self.error:
                true_num = true_num + 1
            else:
                false_num = false_num + 1
        print(true_num)
        print(false_num)
        self.assertGreaterEqual(true_num / (true_num + false_num), self.num_percent)
      

        
    def test_result_two_point(self):
       pass

    def test_result_three_point(self):
       pass
 
    def test_trend(self):
        """
        increase or decrease
        """
        pass


    def test_obj_number(self):
        pass

    def test_run_time(self):
        pass

    def test_write_result(self):
        pass

if __name__ == "__main__":
    unittest.main()
