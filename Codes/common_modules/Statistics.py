import torch
import math

def find_angle_from_coord(coordinates_list,IScrosswalk_list):
    """
    Find the crosswalk angle to the vertical. Inputs: List of crosswalk coordinates, List of crosswalk coordinates considered (IScrosswalk variable)
    """
    thetas = []
    for coordinates,IScrw in zip(coordinates_list,IScrosswalk_list):
        if IScrw:
            theta_x = math.atan((coordinates[3]-coordinates[1])/(coordinates[0]-coordinates[2]))*180/3.14159
            if coordinates[0]>coordinates[2]: thetas.append(90-theta_x)
            else: thetas.append((90+theta_x)*-1)
    return thetas


def find_crosswalk_len(coordinates_list,IScrosswalk_list):
    """
    Find the crosswalk length. Inputs: List of crosswalk coordinates, List of crosswalk coordinates considered (IScrosswalk variable)
    """
    len_arr = []
    for coordinates,IScrw in zip(coordinates_list,IScrosswalk_list):
        if IScrw: len_arr.append(float(((coordinates[3]-coordinates[1])**2+(coordinates[0]-coordinates[2])**2)**(1/2)))
    return len_arr


def mean_distance(Mcoord,Tcoord,Tcrw):
    """
    Find the mean distance and STD between the predicted and ground-truth coordinates. Inputs: Model coordinates, Ground-truth coordinates, list of coordinates considered (IScrosswalk variable); Outputs: Start-point mean distance, Start-point STD, Endpoint mean distance, Endpoint STD
    """
    start_vec,end_vec = [],[]
    for [Mx1,My1,Mx2,My2],[Tx1,Ty1,Tx2,Ty2],IScrw in zip(Mcoord,Tcoord,Tcrw):
        if IScrw != 0:
            start_vec.append(float(((Tx1-Mx1)**2+(Ty1-My1)**2)**0.5))
            end_vec.append(float(((Tx2-Mx2)**2+(Ty2-My2)**2)**0.5))

    start_dist= sum(start_vec)/len(start_vec)
    end_dist  = sum(end_vec)/len(end_vec)

    start_std = ((sum([(Di-start_dist)**2 for Di in start_vec]))/len(start_vec))**0.5
    end_std   = ((sum([(Di-end_dist  )**2 for Di in end_vec  ]))/len(end_vec  ))**0.5

    return start_dist,start_std,end_dist,end_std


def confusion_matrix(model_classes,true_classes,n_classes):
    """
    Create a confusion matrix. Inputs: Model classes, Ground-truth classes, number of classes
    """
    assert n_classes >= 2, 'At least 2 classes are needed'
    M = torch.FloatTensor([[0]*n_classes]*n_classes)
    for Mc,Tc in zip(model_classes,true_classes): M[int(Tc)][round(float(Mc))] += 1
    return M


def class_accuracy(confusion_matrix):
    """
    Calculate the accuracy for one classification task (PTL or crosswalk presence). Inputs: Confusion matrix
    """
    correct_pred = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    return correct_pred/sum(sum(x) for x in confusion_matrix)


def overall_accuracy(Mi,Ti,Ml,Tl):
    """
    Calculate the multi-class accuracy (PTL and crosswalk presence). Inputs: Model IScrosswalk, Ground-truth IScrosswalk, Model Light-class, Ground-truth Light-class
    """
    correct_pred = 0
    for mi,ti,ml,tl in zip(Mi,Ti,Ml,Tl):
        if round(float(mi)) == int(ti) and round(float(ml)) == int(tl): correct_pred += 1
    return correct_pred/len(Ti)


def precision_recall(confusion_matrix):
    """
    Calculate the precision and recall for one classification task (PTL or crosswalk presence). Inputs: Confusion matrix
    """
    precision = []
    recall = []
    for i in range(len(confusion_matrix)):
        col_sum = sum(confusion_matrix[i][:])
        row_sum = sum([row[i] for row in confusion_matrix])

        if row_sum: precision.append(float(confusion_matrix[i][i]/row_sum))
        else: precision.append(0.0)
        if col_sum: recall.append(float(confusion_matrix[i][i]/col_sum))
        else: recall.append(0.0)

    if len(confusion_matrix) == 2: return precision[1],recall[1]
    else: return precision,recall
