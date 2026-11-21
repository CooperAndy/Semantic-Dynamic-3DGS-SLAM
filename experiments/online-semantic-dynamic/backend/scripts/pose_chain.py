import numpy as np
def predict(previous_nominal,current_nominal,previous_selected,has_feedback):
 current=np.asarray(current_nominal,np.float64)
 if not has_feedback:return current.copy()
 return current@np.linalg.inv(np.asarray(previous_nominal,np.float64))@np.asarray(previous_selected,np.float64)
def select(prediction,candidate,accepted):
 return np.asarray(candidate if accepted else prediction,np.float64).copy()
