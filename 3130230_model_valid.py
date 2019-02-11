import numpy as np
import manipulator as mp
import classifiers as classy
kaggle = 0
if(kaggle == 1):
    df_train, df_test, y_train, y_test = mp.kaggletest()
else:
    df_train, df_test, y_train, y_test = mp.localtest()

X_train, X_test = mp.masterManipulator(df_train, df_test, kaggle)

y_train = np.ravel(y_train)
classy.logRes(X_train, X_test, y_train, y_test, kaggle)
classy.sgdClass(X_train, X_test, y_train, y_test, kaggle)
classy.supportVM(X_train, X_test, y_train, y_test, kaggle)
classy.randFost(X_train, X_test, y_train, y_test, kaggle)
# p3130230_model.testrandfo(X_train, X_test, y_train, y_test, kaggle)

'''
0 : 0.5589887640449438
5 : 0.5455056179775281le
'''




'''
comb1
117
0.5128089887640449
0.41213483146067414
0.5613483146067416
0.5231460674157303

comb2
71
0.5065168539325843
0.4395505617977528
0.5006741573033708
0.4997752808988764

comb3
102
0.5060674157303371
0.4404494382022472
0.5173033707865169
0.4813483146067416

comb4
83
0.5087640449438202
0.4404494382022472
0.5253932584269663
0.470561797752809

comb5
114
0.5146067415730337
0.43550561797752807
0.5550561797752809
0.5155056179775281

comb6
86
0.5096629213483146
0.3941573033707865
0.5384269662921348
0.491685393258427

'''
