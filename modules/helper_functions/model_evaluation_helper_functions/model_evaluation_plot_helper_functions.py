import matplotlib.pyplot as plt

def generate_heatmap_from_confusion_matrix(confusion_matrix, class_names):
    '''
    Generate a heatmap from a confuction matrix produced by sklearn.metrics.confusion_matrix
    '''
    
    # Set the figure size
    plt.figure(figsize=(10, 10))

    # Plot the heatmap
    plt.imshow(confusion_matrix, cmap='Blues')

    # Set the title
    plt.title('Confusion Matrix')

    # Set the x and y labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    


