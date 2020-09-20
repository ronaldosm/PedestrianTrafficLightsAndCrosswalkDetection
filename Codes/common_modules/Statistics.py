import torch
import math

def find_angle_from_coord(coordinates_list,IScrosswalk_list):
    thetas = []
    for coordinates,IScrw in zip(coordinates_list,IScrosswalk_list):
        if IScrw: thetas.append(math.atan2(coordinates[3]-coordinates[1],coordinates[0]-coordinates[2])*180/3.14159)
    return thetas


def find_crosswalk_len(coordinates_list,IScrosswalk_list):
    len_arr = []
    for coordinates,IScrw in zip(coordinates_list,IScrosswalk_list):
        if IScrw: len_arr.append(float(((coordinates[3]-coordinates[1])**2+(coordinates[0]-coordinates[2])**2)**(1/2)))
    return len_arr


def EQM(model_coordinates,true_coordinates,true_IScrosswalk):
    start_error, end_error = 0.0, 0.0
    for [Mx1,My1,Mx2,My2],[Tx1,Ty1,Tx2,Ty2],IScrw in zip(model_coordinates,true_coordinates,true_IScrosswalk):
        if IScrw:
            start_error += ((Tx1-Mx1)**2+(Ty1-My1)**2)**0.5
            end_error += ((Tx2-Mx2)**2+(Ty2-My2)**2)**0.5
    div = sum(true_IScrosswalk)
    start_error = start_error/div if div != 0 else 0
    end_error = end_error/div if div != 0 else 0

    return start_error,end_error


def EQM_Array(model_coordinates,true_coordinates,true_IScrosswalk):
    start_error, end_error = [], []
    for [Mx1,My1,Mx2,My2],[Tx1,Ty1,Tx2,Ty2],IScrw in zip(model_coordinates,true_coordinates,true_IScrosswalk):
        if IScrw:
            start_error.append(float(((Tx1-Mx1)**2+(Ty1-My1)**2)**0.5))
            end_error.append(float(((Tx2-Mx2)**2+(Ty2-My2)**2)**0.5))
    return start_error,end_error


def confusion_matrix(model_classes,true_classes,n_classes):
    assert n_classes >= 2, 'At least 2 classes are needed'
    M = torch.FloatTensor([[0]*n_classes]*n_classes)
    for Mc,Tc in zip(model_classes,true_classes): M[int(Tc)][round(float(Mc))] += 1
    return M


def precision_recall(confusion_matrix):
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
