from collections import Counter
import pandas as pd
import numpy as np
import pathlib as path
import os
import glob

##############################################
#---------------------------------------------
# モデルのリビジョン番号
#---------------------------------------------
##############################################
def rev(tsuika,data_mei,model_mei):
    # tsuikaの分解　　　　　
    if tsuika != None:
        m = tsuika.split("__")[0]
        d = tsuika.split("__")[1]
        r = tsuika.split("__")[2]
        e = tsuika.split("__")[3]

    #    param_data = "../kekka/"+m+"__"+d+"__"+r+"/"+tsuika+".hdf5"
    # supermicroではコメントアウトを外す
    #os.chdir("./py_code")
    ##############################################

    print("getcwd()  :",os.getcwd())
    # 追加学習ではない時
    if tsuika == None:
        print("新規学習　だよ")
        revision_name = model_mei+'__'+data_mei+'__'
        kekka = path.Path("Solution1/kekka/")
        print("kekka : ",kekka)
        print("revision_name : ",revision_name)
        already = list(kekka.glob(revision_name + "*/"))
        print("already: ",str(already))
        # already2 = os.listdir("Solution1/kekka/")
        # print("already2: ",already2)

        if not already:
            rev = str(0)
        # co =0
        # for i in range(0,len(already)):
            # if revision_name in str(already[i]):
                # co =co+1
        # print("既存のモデル数　：　",co)
        else:
            # co_x = [str(i).split('_') for i in already]
            # print("co_x : ",co_x)
            co_s = np.array([str(i).split('_') for i in already])[:,-1]
            print("co_s : ",co_s)
            co_int = [int(i) for i in co_s]
            max_co = max(co_int)
            # if co_s:
                # max_co = max(co_s)
            # else:
                # max_co = 0.0
            co = int(max_co)+1
            rev = str(co)

        revision = revision_name+rev

        saki = "./Solution1/kekka/"+revision
        print("model保存先　：　",saki)
        os.mkdir(saki)
        param_data = None
        return(saki,revision,param_data)
    # 追加学習の時
    else :
        print("追加学習")
        revision_name = '__' + d
    #     print('revision_name = ',revision_name)
        param_name = m + '__' + d + '__' + r
    #     print('param_name = ',param_name)
        kekka = Path("./Solution1/kekka/")
    #     print('kekka = ',kekka)

        re_name = "/" + param_name
    #     print('re_name = ',re_name)
        already = list(kekka.glob("**/"+param_name))
    #     print('already = ',already)
        # already_num = len(list(kekka.glob("**/"+param_name+"_?")))+len(list(kekka.glob("**/"+param_name+"_??")))
        already_num = max(int(already.split('_')[-1]))
    #     print('already_num = ',already_num)

        rev = str(int(already_num)+1)
    #     print('rev = ',rev)

        revision = m + revision_name + '__' + r + '_' + rev
    #     print('revision = ',revision)

        saki = str(already[0])+"/"+revision
    #     print('saki = ',saki)
        os.mkdir(str(saki))
        print("model保存先",saki)

        param_data = list(kekka.glob("**/"+tsuika))

        print('param_data =',param_data)
        return(saki,revision,param_data)
