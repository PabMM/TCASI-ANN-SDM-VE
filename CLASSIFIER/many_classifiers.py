#%%Species Classifier using different classifiers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import time #To measure fitting time
from joblib import dump
import sys
sys.path.append('Lib')
from Lib import calculate_confusion_matrix
from Lib import save_model

#%%datareading and normalizatino
csv_file="DATA-SETS/data_classifier_total.csv"
dataframe = pd.read_csv(csv_file)

def print_hash():
  print("########################################")
def print_bar():
  print("________________________________________")




modelsList=dataframe["category"].unique().tolist()
Nmodels=len(modelsList)


print_hash()
print(f'{Nmodels} models')
print_bar()


for sp in modelsList:
    print(f'Model name: {sp}')
print_hash()
#Explicitily creating Species code as target

dataframe["target"]=dataframe["category"]

#Removing variables which will not be used by the classifier

#VarsToRemove=["FemaleID","FemaleCode","Species","SpeciesCode","Ubicacion","UbicationCode","EGG_ID","Orientation","AvgSpotSize","SpotsRSTD","SpotsGSTD","SpotsBSTD","BackGroundRSTD","BackGroundGSTD","BackGroundBSTD"]
VarsToRemove=["category"]

dataframe=dataframe.drop(columns=VarsToRemove)


Column_Names=dataframe.columns.to_list()

Column_Names.remove("target")

for var in Column_Names:
    print(f'Variable Name: {var}')
print_bar()
if(input("Perform minmax scaling? [1 for YES]")=="1"):
  scaler=MinMaxScaler()
  scaled_values=scaler.fit_transform(dataframe[Column_Names])
  dataframe[Column_Names]=scaled_values
  dump(scaler,'CLASSIFIER/model/classifier_scaler.gz')
print_bar()
print(dataframe.head(5))
print_bar()

#split original DataFrame into training and testing sets
train, test = train_test_split(dataframe, test_size=0.2, random_state=0)

x_train=train[Column_Names]
Ninputs= len(Column_Names)

y_train=train["target"]
Noutputs=1

print(f'All models will be trained for {Ninputs} input(s) and {Noutputs} output(s)')
x_test=test[Column_Names]
y_test=test["target"]

x=pd.concat((x_train,x_test),axis=0,ignore_index=True)
y=pd.concat((y_train,y_test),axis=0,ignore_index=True)

# Init calculate confusion Matrix

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(dataframe["target"].values)

CCM = calculate_confusion_matrix(le.classes_)

#%%# First the GNB model
from sklearn.naive_bayes import GaussianNB

GNBmodel = GaussianNB()
tstart=time.time()
GNBmodel.fit(x_train, y_train)
tend=time.time()
ETA=tend-tstart
#Evaluate model
predictions_Test_GNB=GNBmodel.predict(x_test)
predictions_GNB=GNBmodel.predict(x)

#evaluate model
scores_GNB = accuracy_score(y, predictions_GNB)
scores_Test_GNB= accuracy_score(y_test, predictions_Test_GNB)
print(f'GNB accuracy on the test dataset : {np.mean(scores_Test_GNB):.3f} fitting time {ETA:.3f}s')  
print(f'GNB accuracy on the whole dataset: {np.mean(scores_GNB):.3f} fitting time {ETA:.3f}s') 
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictions_Test_GNB,'GNB','Test')
CCM.plot_confusion_matrix(y,predictions_GNB,'GNB','total')

# save model
save_model(GNBmodel,'GNB','CLASSIFIER/model/')

#%% Now the logisticRegresion model
from sklearn import linear_model

classifier_name='logisticRegresion'
classifier_logreg = linear_model.LogisticRegression(max_iter=1000, random_state=0,penalty='elasticnet',solver='saga',l1_ratio=0.5)
tstart=time.time()
classifier_logreg.fit(x_train, y_train)
tend=time.time()
ETA=tend-tstart
#Evaluate model
predictions_Test_logreg=classifier_logreg.predict(x_test)
predictions_logreg=classifier_logreg.predict(x)

#evaluate model
scores_logreg = accuracy_score(y, predictions_logreg)
scores_Test_logreg= accuracy_score(y_test, predictions_Test_logreg)
print(f'{classifier_name} accuracy on the test dataset : {np.mean(scores_Test_logreg):.3f} fitting time {ETA:.3f}s')  
print(f'{classifier_name} accuracy on the whole dataset: {np.mean(scores_logreg):.3f} fitting time {ETA:.3f}s')  
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictions_Test_logreg,'LogReg','Test')
CCM.plot_confusion_matrix(y,predictions_logreg,'LogReg','total')

# save model
save_model(classifier_logreg,'logreg','CLASSIFIER/model/')

#%% Now the MultinomialNB model
from sklearn.naive_bayes import MultinomialNB

classifier_MNB = MultinomialNB()
tstart=time.time()
classifier_MNB.fit(x_train, y_train)
tend=time.time()
ETA=tend-tstart
#Evaluate model
predictions_Test_MNB=classifier_MNB.predict(x_test)
predictions_MNB=classifier_MNB.predict(x)

#evaluate model
scores_MNB = accuracy_score(y, predictions_MNB)
scores_Test_MNB= accuracy_score(y_test, predictions_Test_MNB)
print(f'MNB accuracy on the test dataset : {np.mean(scores_Test_MNB):.3f} fitting time {ETA:.3f}s')  
print(f'MNB accuracy on the whole dataset: {np.mean(scores_MNB):.3f} fitting time {ETA:.3f}s')  
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictions_Test_MNB,'MultinomialNB','Test')
CCM.plot_confusion_matrix(y,predictions_MNB,'MultinomialNB','total')

# save model
save_model(classifier_MNB,'MNB','CLASSIFIER/model/')

#%% First the QDA model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifier_name='Quadratic Discriminant Analysis'
classifier_QDA = QuadraticDiscriminantAnalysis(reg_param=0.6)
tstart=time.time()
classifier_QDA.fit(x_train, y_train)
tend=time.time()
ETA=tend-tstart
#Evaluate model
predictions_Test_QDA=classifier_QDA.predict(x_test)
predictions_QDA=classifier_QDA.predict(x)

#evaluate model
scores_QDA = accuracy_score(y, predictions_QDA)
scores_Test_QDA= accuracy_score(y_test, predictions_Test_QDA)
print(f'{classifier_name} accuracy on the test dataset : {np.mean(scores_Test_QDA):.3f} fitting time {ETA:.3f}s')  
print(f'{classifier_name} accuracy on the whole dataset: {np.mean(scores_QDA):.3f} fitting time {ETA:.3f}s')  
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictions_Test_QDA,'QDA','Test')
CCM.plot_confusion_matrix(y,predictions_QDA,'QDA','total')

# save model
save_model(classifier_QDA,'QDA','CLASSIFIER/model/')

#%% Now the Random Forest model
from sklearn.ensemble import RandomForestClassifier

classifier_name='Random Forest'
classifier_RF = RandomForestClassifier(n_estimators=25, random_state=42,max_depth=32)
tstart=time.time()
classifier_RF.fit(x_train, y_train)
tend=time.time()
ETA=tend-tstart
#Evaluate model
predictions_Test_RF=classifier_RF.predict(x_test)
predictions_RF=classifier_RF.predict(x)

#evaluate model
scores_RF = accuracy_score(y, predictions_RF)
scores_Test_RF= accuracy_score(y_test, predictions_Test_RF)
print(f'{classifier_name} accuracy on the test dataset : {np.mean(scores_Test_RF):.3f} fitting time {ETA:.3f}s')  
print(f'{classifier_name} accuracy on the whole dataset: {np.mean(scores_RF):.3f} fitting time {ETA:.3f}s')  
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictions_Test_RF,'RF','Test')
CCM.plot_confusion_matrix(y,predictions_RF,'RF','total')

# save model
save_model(classifier_RF,'RF','CLASSIFIER/model/')

#%% Now the LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

LDAmodel = LinearDiscriminantAnalysis()
tstart=time.time()
LDAmodel.fit(x_train, y_train)
tend=time.time()
LDATime=tend-tstart
#Evaluate model
predictionsLDA_test=LDAmodel.predict(x_test)
predictionsLDA=LDAmodel.predict(x)

#evaluate model
scoresLDA_test = accuracy_score(y_test, predictionsLDA_test)
scoresLDA = accuracy_score(y, predictionsLDA)
print(f'LDA accuracy on the test data: {np.mean(scoresLDA_test):.3f} execution time {LDATime:.3f}s')   
print(f'LDA accuracy on the whole data: {np.mean(scoresLDA):.3f} execution time {LDATime:.3f}s')   
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictionsLDA_test,'LDA','Test')
CCM.plot_confusion_matrix(y,predictionsLDA,'LDA','total')

# save model
save_model(LDAmodel,'LDA','CLASSIFIER/model/')

#%% ######Now decission tree
from sklearn.tree import DecisionTreeClassifier

DTmodel = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=None, min_samples_leaf=1)   
tstart=time.time()

DTmodel.fit(x_train,y_train)
tend=time.time()
DTTime=tend-tstart

predictionsDT_test=DTmodel.predict(x_test)
predictionsDT=DTmodel.predict(x)

scoresDT_test = accuracy_score(y_test, predictionsDT_test)
scoresDT = accuracy_score(y, predictionsDT)
print(f'Decission Tree accuracy on the test data: {np.mean(scoresDT_test):.3f} execution time {DTTime:.3f}s')   
print(f'Decission Tree accuracy on the whole data: {np.mean(scoresDT):.3f} execution time {DTTime:.3f}s')   
print_bar()

# confusion matrix
CCM.plot_confusion_matrix(y_test,predictionsDT_test,'DT','Test')
CCM.plot_confusion_matrix(y,predictionsDT,'DT','total')

# save model
save_model(DTmodel,'DT','CLASSIFIER/model/')

#%% ###### Now using SVM
from sklearn import svm
SVM_time=[]

for kernel in ("linear", "poly", "rbf"):
  SVMModel = svm.SVC(kernel=kernel, gamma=2)

  #Train the model using the training sets
  tstart=time.time()
  SVMModel.fit(x_train,y_train)
  tend=time.time()
  ThisTime=tend-tstart
  SVM_time.append(ThisTime)
  predictionsSVM_test=SVMModel.predict(x_test)
  predictionsSVM=SVMModel.predict(x)
  scoresSVM_test = accuracy_score(y_test, predictionsSVM_test)
  scoresSVM = accuracy_score(y, predictionsSVM)
  print(f'SVM {kernel} Kernel accuracy on the test data: {np.mean(scoresSVM_test):.3f} execution time {ThisTime:.3f}s')   
  print(f'SVM {kernel} Kernel accuracy on the whole data: {np.mean(scoresSVM):.3f} execution time {ThisTime:.3f}s')
  # confusion matrix
  CCM.plot_confusion_matrix(y_test,predictionsSVM_test,'SVM_'+kernel,'Test')
  CCM.plot_confusion_matrix(y,predictionsSVM,'SVM_'+kernel,'total')

  # save model
  save_model(SVMModel,'SVM_'+kernel,'CLASSIFIER/model/')   

print_bar()

#%% Gradient boosting algorithm
from sklearn.ensemble import GradientBoostingClassifier
#lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
if(input("Adjust learning rate in Gradient Boosting Classifier or use default value?(1-Yes)")=="1"):
  lr_list=np.linspace(0.02,1,40)
  GB_time=[]
  GB_accuracy=[]
  GB_model =[]
  
  for learning_rate in lr_list:
    GBModel = GradientBoostingClassifier(n_estimators=100,learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    tstart=time.time()
    GBModel.fit(x_train, y_train)
    tend=time.time()
    ThisTime=tend-tstart
    predictionsGB=GBModel.predict(x_test)
    scoresGB = np.mean(accuracy_score(y_test, predictionsGB))
    print(f'Learning rate {learning_rate}, accuracy {scoresGB}')
    GB_model.append(GBModel)
    GB_time.append(ThisTime)
    GB_accuracy.append(scoresGB)


#Finding best performing GB model
  BestPerformingGB=np.argmax(GB_accuracy)
  BestLearningRateGB=lr_list[BestPerformingGB]
  BestAccuracyGB=GB_accuracy[BestPerformingGB]
  BestGBTime=GB_time[BestPerformingGB]
  BestModelGB = GB_model[BestPerformingGB]
  # Evaluate best model for metrics and displays
  predictionsGB_test=GBModel.predict(x_test)
  predictionsGB=GBModel.predict(x)
  scoresGB_test = np.mean(accuracy_score(y_test, predictionsGB_test))
  scoresGB = np.mean(accuracy_score(y, predictionsGB))

  print(f'Best performing Gradient Boosting Classifier on test data: {BestAccuracyGB:.3f} using LR={BestLearningRateGB:.2f} execution time {BestGBTime:.3f}s')   
  print(f'Best performing Gradient Boosting Classifier on whole data: {scoresGB:.3f} using LR={BestLearningRateGB:.2f} execution time {BestGBTime:.3f}s')   
  print_bar()

  # confusion matrix
  CCM.plot_confusion_matrix(y_test,predictionsGB_test,'GB','Test')
  CCM.plot_confusion_matrix(y,predictionsGB,'GB','total')

  # save model
  save_model(BestModelGB,'GB','CLASSIFIER/model/')

else:
  GBModel = GradientBoostingClassifier(n_estimators=100,learning_rate=0.52, max_features=2, max_depth=2, random_state=0)
  tstart=time.time()
  GBModel.fit(x_train, y_train)
  tend=time.time()
  ThisTime=tend-tstart
  predictionsGB=GBModel.predict(x_test)
  scoresGB = np.mean(accuracy_score(y_test, predictionsGB))
  print(f' Gradient Boosting Classifier accuracy on the test data: {scoresGB:.3f} execution time {ThisTime:.3f}s')    
  print_bar()

#%% Now using tensorflow
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# Split the dataframe into inputs (X) and target (y)
X = dataframe.drop('target', axis=1)
y = dataframe['target']
# Convert target to integers

y = le.transform(y.values)

# Convert the target column into a one-hot encoded representation
y = tf.keras.utils.to_categorical(y, num_classes=Nmodels)

# Split the inputs and target into training (80%) and test (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert the training and test sets into TensorFlow datasets

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test))
batch_size=64
# Batch and shuffle the training set
train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)

# Batch the test set
test_dataset = test_dataset.batch(batch_size)

# Create a simple model using the inputs and target (architecture is the result of keras NAS tuner)
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(238, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(Nmodels, activation="softmax")
])
print(model.summary())
# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])

# Train the model
history = model.fit(train_dataset, epochs=1, verbose=0, callbacks=[tf.keras.callbacks.History()])

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Tensorflow Network trained with {test_accuracy:.2f} accuracy on the test data')


# Make predictions 
y_pred = model.predict(X)
y_pred_test = model.predict(X_test)
# Convert predictions to a binary format
y_pred_binary = np.round(y_pred)
y_pred_binary_test = np.round(y_pred_test)

Y_test_integer = np.argmax(y, axis=1)
Y_test_integer_test = np.argmax(Y_test, axis=1)
y_pred_binary_integer = np.argmax(y_pred_binary, axis=1)
y_pred_binary_integer_test = np.argmax(y_pred_binary_test, axis=1)




CCM.plot_confusion_matrix(Y_test_integer_test,y_pred_binary_integer_test,'ANN','Test')
CCM.plot_confusion_matrix(Y_test_integer,y_pred_binary_integer,'ANN','Total')
