from sklearn.cluster import KMeans
import numpy as np
from src import autoencoder
import matplotlib.pyplot as plt




#print('---------------------------------------------- auxiliary functions -----------------------------------------------------------')
# recommendation
def binary_to_dec(recommendering_user):
    dec_array = []
    for i in range(636):
        first = i * 5
        second = i * 5 + 1
        third = i * 5 + 2
        fourth = i * 5 + 3
        fifth = i * 5 + 4

        value = recommendering_user[first] * (2 ** 4) + recommendering_user[second] * (2 ** 3) + \
                recommendering_user[third] * (2 ** 2) + recommendering_user[fourth] * (2 ** 1) + recommendering_user[fifth]

        dec_array.append(value)

    return dec_array

def binary_matrix_to_dec_matrix(recommendering_user):
    result = []
    for i in range(len(recommendering_user)):
        result.append( binary_to_dec(recommendering_user[i]) )
    return result

def loss_function_accuracy_difference_between_true_label_and_probability(training_data_accuracy,true_label):
    if true_label == 16 :
        normalised_true_label = 0
    else:
        normalised_true_label = 1
    result = np.abs((training_data_accuracy-normalised_true_label))
    if result>1:
        print('---caesar---')
    return result

def loss_function_accuracy_difference_between_true_label_and_prediction(training_data_accuracy,true_label):
    if true_label == 16 :
        normalised_true_label = 0
    else:
        normalised_true_label = 1

    if training_data_accuracy>0.5:
        predicted_value = 1
    else:
        predicted_value = 0
    result = np.abs((normalised_true_label-predicted_value))
    return result

def fill_the_clusters(label_pred):
    clusters = {}
    for index, cluster_index in enumerate(label_pred):
        if cluster_index not in clusters.keys():
            clusters[cluster_index] = [ original_training_data[index] ]
        else:
            clusters[cluster_index].append(original_training_data[index])
    return clusters

def run_kmeans(n_clusters):
    clusters = {}
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(compressed_training_data)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    centroids = np.asarray(centroids)

    return estimator,label_pred,centroids

def calculate_loss(clusters):

    loss_difference_between_true_label_and_probability = []

    for i in range( len(compressed_testing_data) ):# for every test case

        label = estimator.predict(np.reshape(compressed_testing_data[i],(1,-1)))

        tempt = np.copy( clusters[label[0]] )

        recommendering_user_binary = np.copy(original_testing_data[i])

        # calculate accuracy
        recommendering_user = binary_to_dec(recommendering_user_binary)

        new_matrix = binary_matrix_to_dec_matrix( tempt )
        #print(np.shape(new_matrix))
        row_index = np.shape(new_matrix)[0]

        poll_num = len(recommendering_user)
        individual_accuracy = []

        for j in range(poll_num): # for every poll of this test user
            true_label =recommendering_user[j]
            count = 0
            true = 0
            training_count = 0
            for k in range(row_index):

                if recommendering_user[j] != 0 :
                    count = count+1
                    new_matrix = np.asarray(new_matrix)
                    column = new_matrix[:,j]

                    if column[k] != 16 and column[k] != 0:
                        true = true + 1
                    if column[k] !=0:
                        training_count = training_count+1

            if training_count != 0:
                training_data_accuracy = true/training_count
                the_loss_of_each_poll = loss_function_accuracy_difference_between_true_label_and_probability(training_data_accuracy,true_label)
                individual_accuracy.append(the_loss_of_each_poll)

        the_loss_of_a_user = np.average(individual_accuracy)
        loss_difference_between_true_label_and_probability.append(the_loss_of_a_user)
    return loss_difference_between_true_label_and_probability

#print('---------------------------------------------- auxiliary functions -----------------------------------------------------------')


a = autoencoder.autoencoder(num_epochs=100, denoising=False, masking=0.5, display_step=200)

compressed_training_data = a.compressed_train

original_training_data = a.train

compressed_testing_data = a.compressed_test
original_testing_data = a.test

#n_clusters =20
train_loss = []
for n_clusters in np.arange(1,11):
    average_loss_with_different_k_values = []

    estimator,label_pred,centroids = run_kmeans(n_clusters)

    clusters = fill_the_clusters(label_pred)

    # print(len(clusters))

    loss_difference_between_true_label_and_probability = calculate_loss(clusters)

    # print('k is :                                               %s' % n_clusters)
    l = np.average(loss_difference_between_true_label_and_probability)
    # print(n_clusters, np.average(loss_difference_between_true_label_and_probability))

    print('loss_difference_between_true_label_and_probability : %s'% np.average(loss_difference_between_true_label_and_probability))
    average_loss_with_different_k_values.append((n_clusters,l))
    train_loss.append(np.average(loss_difference_between_true_label_and_probability))

xs = np.arange(1,11)
ys = train_loss
fig, ax = plt.subplots(figsize=(8, 4))
ys = np.reshape(ys,(10,-1))
ax.plot(xs, ys)
ax.set_xlabel('the value of k')
ax.set_ylabel('the loss of recommender system, difference between recommendation and reality')
plt.show()

print('----------------------------- the execution ends here -----------------------------------------')

