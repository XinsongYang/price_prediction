# Field Name        Definition
# RefID               Unique (sequential) number assigned to vehicles
# IsBadBuy        Identifies if the kicked vehicle was an avoidable purchase 
# PurchDate       The Date the vehicle was Purchased at Auction
# Auction         Auction provider at which the  vehicle was purchased
# VehYear         The manufacturer's year of the vehicle
# VehicleAge        The Years elapsed since the manufacturer's year
# Make          Vehicle Manufacturer 
# Model         Vehicle Model
# Trim          Vehicle Trim Level
# SubModel        Vehicle Submodel
# Color         Vehicle Color
# Transmission        Vehicles transmission type (Automatic, Manual)
# WheelTypeID       The type id of the vehicle wheel
# WheelType       The vehicle wheel type description (Alloy, Covers)
# VehOdo          The vehicles odometer reading
# Nationality       The Manufacturer's country
# Size          The size category of the vehicle (Compact, SUV, etc.)
# TopThreeAmericanName      Identifies if the manufacturer is one of the top three American manufacturers
# MMRAcquisitionAuctionAveragePrice Acquisition price for this vehicle in average condition at time of purchase 
# MMRAcquisitionAuctionCleanPrice   Acquisition price for this vehicle in the above Average condition at time of purchase
# MMRAcquisitionRetailAveragePrice  Acquisition price for this vehicle in the retail market in average condition at time of purchase
# MMRAcquisitonRetailCleanPrice   Acquisition price for this vehicle in the retail market in above average condition at time of purchase
# MMRCurrentAuctionAveragePrice   Acquisition price for this vehicle in average condition as of current day 
# MMRCurrentAuctionCleanPrice   Acquisition price for this vehicle in the above condition as of current day
# MMRCurrentRetailAveragePrice    Acquisition price for this vehicle in the retail market in average condition as of current day
# MMRCurrentRetailCleanPrice    Acquisition price for this vehicle in the retail market in above average condition as of current day
# PRIMEUNIT       Identifies if the vehicle would have a higher demand than a standard purchase
# AcquisitionType       Identifies how the vehicle was aquired (Auction buy, trade in, etc)
# AUCGUART        The level guarntee provided by auction for the vehicle (Green light - Guaranteed/arbitratable, Yellow Light - caution/issue, red light - sold as is)
# KickDate        Date the vehicle was kicked back to the auction
# BYRNO         Unique number assigned to the buyer that purchased the vehicle
# VNZIP                                   Zipcode where the car was purchased
# VNST                                    State where the the car was purchased
# VehBCost        Acquisition cost paid for the vehicle at time of purchase
# IsOnlineSale        Identifies if the vehicle was originally purchased online
# WarrantyCost                            Warranty price (term=36month  and millage=36K) 




import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm

import time

class LemonCarFeaturizer():
  def __init__(self):
    vectorizer = None
    self._selection = SelectKBest(chi2, k=8)
    self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    self._binarizer = preprocessing.Binarizer()
    self._scaler = preprocessing.MinMaxScaler()
    self._preprocs = [self._imputer, \
                      # self._binarizer, \
                      self._scaler
                      ]

  def _fit_transform(self, dataset):
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    for p in self._preprocs:
      dataset = p.transform(dataset)

    return dataset

  def _proc_fit_transform(self, p, dataset):
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset

  def create_features(self, dataset, training=False):
    data = dataset[ [
                  # 'MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentAuctionAveragePrice',
                  'MMRCurrentAuctionCleanPrice',
                  'MMRCurrentRetailAveragePrice',
                  'VehYear',
                  'VehicleAge',
                  'VehOdo',
                  'PurchDate',
                  # 'KickDate',
                  'WarrantyCost',
                  # 'MMRAcquisitionAuctionAveragePrice',
                  # 'MMRAcquisitionAuctionCleanPrice',
                  # 'MMRAcquisitionRetailAveragePrice',
                  # 'MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentRetailCleanPrice',
                  'VehBCost'
                  ]
          ]
    #data ['']
    dummy_features = ['Auction', 'TopThreeAmericanName', 'AUCGUART', 'Transmission', 'WheelTypeID', 'WheelType',  'Size']
    # dummy_features = ['Auction', 'Size']
    for f in dummy_features:
        dummy_data = pd.get_dummies(dataset[f])
        data = data.join(dummy_data, rsuffix=f)
        
    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data

  def create_test_features(self, train_dataset, test_dataset):
    data = test_dataset[ [
                  # 'MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentAuctionAveragePrice',
                  'MMRCurrentAuctionCleanPrice',
                  'MMRCurrentRetailAveragePrice',
                  'VehYear',
                  'VehicleAge',
                  'VehOdo',
                  'PurchDate',
                  # 'KickDate',
                  'WarrantyCost',
                  # 'MMRAcquisitionAuctionAveragePrice',
                  # 'MMRAcquisitionAuctionCleanPrice',
                  # 'MMRAcquisitionRetailAveragePrice',
                  # 'MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentRetailCleanPrice',
                  'VehBCost'
                  ]
          ]
    #data ['']
    dummy_features = ['Auction', 'TopThreeAmericanName', 'AUCGUART', 'Transmission', 'WheelTypeID', 'WheelType',  'Size']
    for f in dummy_features:
      train_dummy = pd.get_dummies(train_dataset[f])
      test_dummy = pd.get_dummies(test_dataset[f])
      for train_f in train_dummy.columns:
        if train_f not in test_dummy.columns:
          test_dummy[train_f] = 0
      data = data.join(test_dummy, rsuffix=f)

    data = self._transform(data)
    return data

def train_model(X, y):
  # model = RidgeClassifierCV()
  model = LogisticRegression(C=10)
  # model = DecisionTreeClassifier() 
  # model = RandomForestClassifier(n_estimators=500)
  # model = svm.LinearSVC()
  model.fit(X, y)
  #print model.coef_
  return model

def predict(model, y):
  return model.predict(y)

def create_submission(model, transformer):
  submission_test = pd.read_csv('inclass_test.csv', converters={'PurchDate':dateToNum})
  train_data = pd.read_csv('inclass_training.csv')
  #print(model.predict_proba(transformer.create_features(submission_test)))
  predictions = pd.Series([x for x in model.predict(transformer.create_test_features(train_data, submission_test))])
  submission = pd.DataFrame({'RefId': submission_test.RefId, 'IsBadBuy': predictions})
  submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission_test.csv', index=False)

def test_feature(X, y):
	X_new = SelectKBest(chi2, k=8).fit_transform(X, y)
	print(X_new)

def dateToNum(date):
  timeArray = time.strptime(date, "%m/%d/%Y")
  return int(time.mktime(timeArray))

def main():
  data = pd.read_csv('inclass_training.csv', converters={'PurchDate':dateToNum})
  featurizer = LemonCarFeaturizer()
  
  print ("Transforming dataset into features...")
  X = featurizer.create_features(data, training=True)
  y = data.IsBadBuy
  # print(X)
  # test_feature(X,y)

  print ("Training model...")
  model = train_model(X,y)

  print ("Cross validating...")
  print (np.mean(cross_val_score(model, X, y, scoring='roc_auc')))

  print ("Create predictions on submission set...")
  create_submission(model, featurizer)


if __name__ == '__main__':
  main()
