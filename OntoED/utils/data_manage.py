# -*- coding: utf-8 -*-
import pickle
import settings.parameters as para


def pickle_save(data, filePath):  # 使用pickle模块将数据对象保存到文件
    f = open(filePath, 'wb')
    pickle.dump(data, f)
    f.close()


def pickle_load(filePath):  # 使用pickle从文件中重构python对象
    f = open(filePath, 'rb')
    data = pickle.load(f)
    # pprint.pprint(data)
    f.close()
    return data

