import numpy as np
import pandas as pd
import cv2

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis Client
hostname = 'redis-15185.c305.ap-south-1-1.ec2.redns.redis-cloud.com'
portnumber = 15185
password = 'vfJ9zVbFFbzPueK4ttiVuANMwsQ0VPrd'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name', 'Role'], thresh=0.5):
    """
    Cosine similarity-based search algorithm
    """
    # Step-1: Take the list of face embeddings from the dataframe
    X_list = dataframe[feature_column].tolist()

    # Step-2: Filter valid embeddings by checking their shape
    valid_rows = [i for i, embedding in enumerate(X_list) if len(embedding) == len(test_vector)]
    
    # If no valid embeddings, return 'Unknown'
    if not valid_rows:
        return 'Unknown', 'Unknown'

    # Filter dataframe and embeddings
    valid_dataframe = dataframe.iloc[valid_rows].copy()
    valid_embeddings = [X_list[i] for i in valid_rows]

    # Step-3: Convert the valid embeddings list into a NumPy array
    x = np.asarray(valid_embeddings)

    # Step-4: Calculate cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()

    # Add similarity scores to the filtered dataframe
    valid_dataframe['cosine'] = similar_arr

    # Step-5: Filter based on the threshold
    data_filter = valid_dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # Step-6: Identify the best match
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role



def face_prediction(test_image,dataframe,feature_column,
                        name_role=['Name','Role'],thresh=0.5):
    # step-1: take the test image and apply to insight face
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    # step-2: use for loop and extract each embedding and pass to ml_search_algorithm
    
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe,
                                                       feature_column,
                                                       test_vector=embeddings,
                                                       name_role=name_role,
                                                       thresh=thresh)
        if person_name == 'Unknown':
            color =(0,0,255) # bgr
        else:
            color = (0,255,0)
            
            
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
        
        text_gen = person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
        
    return test_copy
