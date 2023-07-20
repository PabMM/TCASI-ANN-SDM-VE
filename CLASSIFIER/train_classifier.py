import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import unique
from numpy import argmax
from numpy import arange
from numpy import newaxis
from numpy import save
from itertools import product
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
import matplotlib.pyplot as plt


# load dataset
model_name = 'classifier'
file_name = 'DATA-SETS/data_'+model_name

df_train = read_csv(file_name+'_train.csv')
df_val = read_csv(file_name+'_val.csv')

# split into input (X) and output (y) variables
X_train = df_train[['SNR','OSR','Power']].values
X_val = df_val[['SNR','OSR','Power']].values

# encode strings to integer
le = LabelEncoder()

y_train = le.fit_transform(df_train['category'].values)
y_val =le.transform(df_val['category'].values)

save('CLASSIFIER/model/'+model_name+'_classes.npy', le.classes_,allow_pickle=True)




num_inputs = X_train.shape[1] #specs should be the same regardless the architecture
n_class = len(unique(y_train))


# model parameters
units = 64
activation = 'relu'
optimizer = 'adam'

# model estructure
input_vector = keras.layers.Input(shape=(num_inputs,))
x = keras.layers.BatchNormalization(scale=True, center=True)(input_vector)
x = keras.layers.Dense(units=units, activation=activation)(x)
x1 = keras.layers.LayerNormalization( scale=True, center=True, axis=-1)(x)
x = keras.layers.Dense(units=units, activation=activation)(x1)
x = keras.layers.LayerNormalization( scale=True, center=True, axis=-1)(x)
x  = keras.layers.concatenate([x1,x]) #skip layer

# classification output
out_clas = keras.layers.Dense(n_class, activation='softmax',name = 'classification')(x)

# define model
model = keras.Model(inputs=input_vector, outputs=out_clas)

# compile the keras model
model.compile(loss=['sparse_categorical_crossentropy'],
           metrics={'classification': 'sparse_categorical_accuracy'},optimizer=optimizer)


# callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=0.00001,restore_best_weights=True,verbose = 1)


checkpoint_filepath = 'CLASSIFIER/model/checkpoint'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=True)
callbacks = [early_stop,model_checkpoint_callback, keras.callbacks.TensorBoard("CLASSIFIER/model/tb_logs")]

model.fit(X_train, y_train,
             validation_data=(X_val,y_val),epochs=2000, batch_size=32,callbacks = callbacks,verbose=2)
model.save('CLASSIFIER/model/'+model_name,overwrite= True)

# print model
print(model.summary())

# plot graph of model
# keras.utils.plot_model(model, to_file='CLASSIFIER/model/'+model_name+'.png', show_shapes=True)


# Define classes
classes = le.classes_

def plot_confusion_matrix(true_class,predict_class,name):
    # Plot confusion matrix
    conf_matrix = confusion_matrix(true_class,predict_class)
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:,newaxis] # normalize here
    plt.imshow(normalized_conf_matrix, cmap=plt.cm.Blues)

    # Add labels
    plt.title("Normalized Confusion Matrix "+name) # change title to reflect normalization
    plt.colorbar()
    tick_marks = arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add values to cells
    thresh = normalized_conf_matrix.max() / 2.
    for i, j in product(range(normalized_conf_matrix.shape[0]), range(normalized_conf_matrix.shape[1])):
        plt.text(j, i, format(normalized_conf_matrix[i, j], '.2f'), # format to display two decimal places
                horizontalalignment="center",
                color="white" if normalized_conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('CLASSIFIER/Confusion_Matrices/Confusion_Matrix_'+model_name+'_'+name+'.png')
    plt.pause(3)
    plt.close()

# make predictions on test set
# eval_results
eval_results = model.evaluate(X_val,y_val,verbose=0)

print('Accuracy val set: %.3f' % eval_results[1])

y_val_predict = model.predict(X_val)
y_val_predict = argmax(y_val_predict, axis=-1).astype('int')
plot_confusion_matrix(y_val,y_val_predict,'Val Set')

# make predictions on total data set
 # eval_results
eval_results = model.evaluate(X_train,y_train,verbose=0)

print('Accuracy val set: %.3f' % eval_results[1])

y_train_predict = model.predict(X_train)
y_train_predict = argmax(y_train_predict, axis=-1).astype('int')
plot_confusion_matrix(y_train,y_train_predict,'Train Set')