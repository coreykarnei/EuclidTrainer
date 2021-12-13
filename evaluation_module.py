import numpy as np

# main function called to evaluate the skeleton's posture
def evaluate_exercise(skeleton, exercise_type):
    print("Evaluating your posture...")
    if exercise_type == "curl":
        corrections = evaluate_curl(skeleton)
    elif exercise_type == "squat":
        corrections = evaluate_squat(skeleton)
    
    if corrections == "":
        print(f"Your {exercise_type} form is quite good! I can't find anything to correct!")
    else:
        print(corrections)
    return

# called to evaluate a squat 
def evaluate_squat(skeleton):
    corrections = ""
    corrections += check_feet_distance(skeleton)
    corrections += check_butt_depth(skeleton)

    return corrections

# called to evaluate a curl
def evaluate_curl(skeleton):
    corrections = ""
    corrections += check_forearm_swing(skeleton) 
    corrections += check_spine_swing(skeleton)

    return corrections

# returns a correction if the forearm swings during a curl
def check_forearm_swing(skeleton):
    body_parts_of_angle = ('A', 'Elbow', 'Shoulder', 'Hip')

    min_ang = skeleton.angle_dict[0][body_parts_of_angle]
    max_ang = skeleton.angle_dict[0][body_parts_of_angle]
    mean_ang = 0
    num_nonnan = 0
    for i, v in enumerate(skeleton.angle_dict):
        if not np.isnan(v[body_parts_of_angle]):
            num_nonnan += 1
            mean_ang += v[body_parts_of_angle]

        if v[body_parts_of_angle] > max_ang:
            max_ang = v[body_parts_of_angle]
        if v[body_parts_of_angle] < min_ang:
            min_ang = v[body_parts_of_angle]

    mean_ang = mean_ang / num_nonnan
    mean_ang = np.degrees(mean_ang)

    if mean_ang > 8:
        return "You are swinging your arm too much! Try to keep your forearm parallel to your spine.\n"
    else:
        return ""

# returns a correction if the person's torso is moving too much 
def check_spine_swing(skeleton):

    body_parts_of_angle = ('A', 'Knee', 'Hip', 'Shoulder')

    angles = list()

    for i, v in enumerate(skeleton.angle_dict):
        if not np.isnan(v[body_parts_of_angle]):
            angles.append(v[body_parts_of_angle])

    angles = np.array(angles)
    angles = skeleton.reject_outliers(angles, m=100)
    angles = np.degrees(angles)
    angle_range = angles.max() - angles.min()

    if angle_range > 20:
        return "Your torso is moving too much! Try and keep it as still as possible."
    else:
        return ""

# returns a correction if the feet are too close together 
def check_feet_distance(skeleton):
    feet_dist = []
    shoulder_dist = []

    for i, v in enumerate(skeleton.feat_dict):
        x1, y1, c1 = v["RAnkle"]
        x2, y2, c2 = v["LAnkle"]
        feet_dist.append(skeleton.edist(x1, y1, x2, y2))
        

        x1, y1, c1 = v["RShoulder"]
        x2, y2, c2 = v["LShoulder"]
        shoulder_dist.append( skeleton.edist(x1, y1, x2, y2))
    feet_dist = np.array(feet_dist)
    shoulder_dist = np.array(shoulder_dist)
    # for i, value in enumerate(feet_dist):
    #     if value == 0:
    #         feet_dist[i] == 0.0001

    ratio = shoulder_dist / (feet_dist + 1e-7)
    if ratio.mean() > 1.3:
        return "Your feet are too close together! Place them about shoulder width apart.\n"
    else:
        return ""

# returns a correction if the butt gets too low
def check_butt_depth(skeleton):
    body_parts_of_angle = ('A', 'Ankle', 'Knee', 'Hip')

    angles = list()

    for i, v in enumerate(skeleton.angle_dict):
        if not np.isnan(v[body_parts_of_angle]):
            angles.append(v[body_parts_of_angle])

    angles = np.array(angles)
    angles = skeleton.reject_outliers(angles, m=4)
    angles = np.degrees(angles)
    butt_range = angles.max() - angles.min()
    if butt_range < 80:
        return "You are squatting too deep! Make sure your butt don't go lower than your knees."
    else:
        return ""