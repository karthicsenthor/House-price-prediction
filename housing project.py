# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:09:01 2020

@author: karthik SenThor
"""



import pandas as pd

dataset=pd.read_csv("D:\BA&DS\machine learning\housing assignment\\train1.csv")
dataset1=pd.read_csv("D:\\BA&DS\\machine learning\\housing assignment\\test1.csv")
x_train=dataset.iloc[:,:-1]
y_train=pd.DataFrame(dataset.iloc[:,-1])
x_test=dataset1.iloc[:,:]
x_train.LotFrontage=x_train['LotFrontage'].fillna(x_train['LotFrontage'].mean())
x_test.LotFrontage=x_test['LotFrontage'].fillna(x_test['LotFrontage'].mean())

res_train=pd.get_dummies(dataset.loc[:,['MSZoning',"Alley","LotShape","LandContour","Utilities","LotConfig",'LandSlope',"BldgType","HouseStyle",
       "RoofStyle","Exterior1st","Exterior2nd","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure",
       "Heating","HeatingQC","CentralAir","Electrical","Functional","GarageType","GarageFinish","GarageQual","PavedDrive",
       "Fence"]])

res_test=pd.get_dummies(dataset1.loc[:,['MSZoning',"Alley","LotShape","LandContour","Utilities","LotConfig",'LandSlope',"BldgType","HouseStyle",
       "RoofStyle","Exterior1st","Exterior2nd","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure",
       "Heating","HeatingQC","CentralAir","Electrical","Functional","GarageType","GarageFinish","GarageQual","PavedDrive",
       "Fence"]])

res1=x_train.loc[:,["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","BsmtUnfSF",
                     "TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","BedroomAbvGr","Fireplaces","GarageArea","WoodDeckSF","OpenPorchSF",
                     "EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","YrSold"]]

res2=x_test.loc[:,["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","BsmtUnfSF",
                     "TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","BedroomAbvGr","Fireplaces","GarageArea","WoodDeckSF","OpenPorchSF",
                     "EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","YrSold"]]
    
x_train1=pd.concat([res_train,res1],axis=1)
x_test1=pd.concat([res_test,res2],axis=1)




def vif_calc(x):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x_train1["intercept"]=1
    
    vif=pd.DataFrame()
    vif["variables"]=x_train1.columns
    vif["vif"]=[variance_inflation_factor(x_train1.values,i)for i in range(0,x_train1.shape[1])]
    return(vif)
    

vif=vif_calc(x_train1)

x_train1=x_train1.drop(["MSZoning_C (all)"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["LotShape_IR1"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["LandContour_Bnk"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Utilities_AllPub"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["LotConfig_Corner"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["LandSlope_Gtl"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["BldgType_1Fam"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["HouseStyle_1.5Fin"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["RoofStyle_Flat"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_AsbShng"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_CBlock"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior2nd_AsbShng"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["ExterQual_Ex"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["ExterCond_Ex"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Foundation_BrkTil"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["BsmtQual_Ex"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Heating_Floor"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["HeatingQC_Ex"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["CentralAir_N"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Functional_Maj1"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageType_2Types"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageFinish_Fin"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageQual_Ex"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["PavedDrive_N"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Fence_GdPrv"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Electrical_FuseA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_VinylSd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["BsmtExposure_No"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageType_Attchd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["ExterCond_TA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior2nd_MetalSd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageQual_TA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["MSSubClass"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["RoofStyle_Hip"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Heating_GasA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_CemntBd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["BsmtCond_TA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["2ndFlrSF"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["MSZoning_RL"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["HouseStyle_1Story"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_BrkFace"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_HdBoard"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_MetalSd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_Plywood"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_Stucco"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior1st_Wd Sdng"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["ExterQual_TA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Exterior2nd_VinylSd"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Foundation_PConc"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["BsmtQual_TA"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Functional_Typ"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Electrical_Mix"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["GarageFinish_Unf"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["Fence_none"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["OverallQual"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["YearBuilt"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["1stFlrSF"],1)
vif=vif_calc(x_train1)
x_train1=x_train1.drop(["TotalBsmtSF"],1)
vif=vif_calc(x_train1)










x_test1=x_test1.drop(["MSZoning_C (all)"],1)
x_test1=x_test1.drop(["LotShape_IR1"],1)
x_test1=x_test1.drop(["LandContour_Bnk"],1)
x_test1=x_test1.drop(["Utilities_AllPub"],1)
x_test1=x_test1.drop(["LotConfig_Corner"],1)
x_test1=x_test1.drop(["LandSlope_Gtl"],1)
x_test1=x_test1.drop(["BldgType_1Fam"],1)
x_test1=x_test1.drop(["HouseStyle_1.5Fin"],1)
x_test1=x_test1.drop(["RoofStyle_Flat"],1)
x_test1=x_test1.drop(["Exterior1st_AsbShng"],1)
x_test1=x_test1.drop(["Exterior1st_CBlock"],1)
x_test1=x_test1.drop(["Exterior2nd_AsbShng"],1)
x_test1=x_test1.drop(["ExterQual_Ex"],1)
x_test1=x_test1.drop(["ExterCond_Ex"],1)
x_test1=x_test1.drop(["Foundation_BrkTil"],1)
x_test1=x_test1.drop(["BsmtQual_Ex"],1)
#x_test1=x_test1.drop(["Heating_Floor"],1)
x_test1=x_test1.drop(["HeatingQC_Ex"],1)
x_test1=x_test1.drop(["CentralAir_N"],1)
x_test1=x_test1.drop(["Functional_Maj1"],1)
x_test1=x_test1.drop(["GarageType_2Types"],1)
x_test1=x_test1.drop(["GarageFinish_Fin"],1)
#x_test1=x_test1.drop(["GarageQual_Ex"],1)
x_test1=x_test1.drop(["PavedDrive_N"],1)
x_test1=x_test1.drop(["Fence_GdPrv"],1)
x_test1=x_test1.drop(["Electrical_FuseA"],1)
x_test1=x_test1.drop(["Exterior1st_VinylSd"],1)
x_test1=x_test1.drop(["BsmtExposure_No"],1)
x_test1=x_test1.drop(["GarageType_Attchd"],1)
x_test1=x_test1.drop(["ExterCond_TA"],1)
x_test1=x_test1.drop(["Exterior2nd_MetalSd"],1)
x_test1=x_test1.drop(["GarageQual_TA"],1)
x_test1=x_test1.drop(["MSSubClass"],1)
x_test1=x_test1.drop(["RoofStyle_Hip"],1)
x_test1=x_test1.drop(["Heating_GasA"],1)
x_test1=x_test1.drop(["Exterior1st_CemntBd"],1)
x_test1=x_test1.drop(["BsmtCond_TA"],1)
x_test1=x_test1.drop(["2ndFlrSF"],1)
x_test1=x_test1.drop(["MSZoning_RL"],1)
x_test1=x_test1.drop(["HouseStyle_1Story"],1)
x_test1=x_test1.drop(["Exterior1st_BrkFace"],1)
x_test1=x_test1.drop(["Exterior1st_HdBoard"],1)
x_test1=x_test1.drop(["Exterior1st_MetalSd"],1)
x_test1=x_test1.drop(["Exterior1st_Plywood"],1)
x_test1=x_test1.drop(["Exterior1st_Stucco"],1)
x_test1=x_test1.drop(["Exterior1st_Wd Sdng"],1)
x_test1=x_test1.drop(["ExterQual_TA"],1)
x_test1=x_test1.drop(["Exterior2nd_VinylSd"],1)
x_test1=x_test1.drop(["Foundation_PConc"],1)
x_test1=x_test1.drop(["BsmtQual_TA"],1)
x_test1=x_test1.drop(["Functional_Typ"],1)
#x_test1=x_test1.drop(["Electrical_Mix"],1)
x_test1=x_test1.drop(["GarageFinish_Unf"],1)
x_test1=x_test1.drop(["Fence_none"],1)
x_test1=x_test1.drop(["OverallQual"],1)
x_test1=x_test1.drop(["YearBuilt"],1)
x_test1=x_test1.drop(["1stFlrSF"],1)
x_test1=x_test1.drop(["TotalBsmtSF"],1)


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
x_train2=x_train1.drop(["Utilities_NoSeWa","HouseStyle_2.5Fin","Exterior1st_ImStucc","Exterior1st_Stone",
                        "Exterior1st_Stone","Heating_OthW","intercept","Exterior2nd_Other"],1)
regression.fit(x_train2,y_train)
regression.coef_
regression.intercept_


x_test1=x_test1.fillna(x_test1.mean())
y_pred=regression.predict(x_test1)

import pandas as pd 
y_pred1=pd.DataFrame(y_pred)
y_pred1.to_csv("housing prediction.csv")









