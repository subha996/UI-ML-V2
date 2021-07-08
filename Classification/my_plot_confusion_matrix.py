
#Create a confusion matrix fuction

def my_confusion_matrix(y_true, y_pred, classes=None, figsize=(7,6), text_size=8):
  '''
  ===============================================================================================================================
  ::
  If creating confusion matrix for binary model(0 and 1) the pass only y_true and y_pred. leave the `classe` argument.
  Please calculate the prediction before passing to the function.
  If the output is prddiction probability(like 'sigmoid function does) then use round() function before passing it to the function.
  ::
  If creating confusion matricx for multiclass then pass the classes as the list of class name.
  Remember the class list index have to match with your prediction call result.
  :: e.g. > class_list = ['A', 'B', 'C', 'D'] 
          Now if our model give us the output of 1 then the final result will be 'B' as 'B' is 2nd index of our above class_list.
          ::>>
          the function will give us >> class_list[1] 
                                    >> 'B'
      Create a list for your class and then pass the list in to the function 'classes' argument.
     ..............................................................................................
  :: Remember classes doesn't matter for this function. 
     This function can work with any model or classifier as long you can pass 'y_true' and 'y_pred'
     ..............................................................................................
  ================================================================================================================================
  '''
  # Importing necessary library
  import itertools
  import numpy as np
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt
  #create a confusion matrix
  cm = confusion_matrix(y_true, y_pred) # first creating a confusion matrix
  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalize our confusion matrix
  n_class = cm.shape[0]
  fig, ax = plt.subplots(figsize=figsize) # setting plot size
  #create matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # #create classes
  # classes = False

  #Set the labels to be class
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])


  #label the axis
  ax.set(title='Confusion Matrix',
        xlabel='Prediction Label',
        ylabel='Actual label',
        xticks=np.arange(n_class),
        yticks=np.arange(n_class),
        xticklabels=labels,
        yticklabels=labels)

  #set X-axis labels to the bottom
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  #adjust the label size
  ax.yaxis.label.set_size(10)
  ax.xaxis.label.set_size(10)
  ax.title.set_size(11)


  #set the threshold for different color

  threshold  = (cm.max() + cm.min()) / 2.

  #plot the text on each cell
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, f'{cm[i,j]} ({cm_norm[i,j]*100:.2f}%)',
                        horizontalalignment='center',
                        color='white' if cm[i,j]>threshold else 'black',
                          size=text_size)