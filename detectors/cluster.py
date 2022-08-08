import rospy
import utils
import tf
import message_filters
from sensor_msgs.msg import LaserScan
import sensor_msgs.point_cloud2 as pc2

lp = lg.LaserProjection()
listener = tf.TransformListener()

tfs = []

def callback(front, left, right, middle):

    points = {}
    msgs = [front, left, right, middle]
    for idx in len(msgs):
        msg = msgs[idx]
        msg = lp.projectLaser(msg)
        msg = pc2.read_points(msg)

        for point in msg:
            intensity = point[3]
            point = np.array([*point[:3], 1])
            point = np.matmul(mat, point)
            point[-1] = intensity
            points[msgs[idx].header.frame_id].append(point)

     for key, value in points.items():
        points[key] = filterRobot(value)

    

def filterRobot(points):
    newPoints = []
    for i in range(len(points)):
        p = points[i]
        if p[0] < 0.25 and p[0] > -1.4:
            if p[1] < 1.5 and p[1] > -1.5:
                continue
        if p[0] < -1.3 and p[0] > -4.8:
            if p[1] < 1.3 and p[1] > -1.3:
                continue
        newPoints.append(p)
    return newPoints
    
def getMat(trans):
    r = R.from_quat(trans[1])
    mat = r.as_matrix()
    mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
    mat[0][-1] += trans[0][0]
    mat[1][-1] += trans[0][1]
    mat[2][-1] += trans[0][2]
    mat[3][-1] = 1
    tfs.append(mat)

if __name__ == "__main__":

    rospy.init_node("clusterer")

    try:
        tf_front = listener.lookupTransform('/base_link', '/sick_front', rospy.Time(0))
        tf_back_left = listener.lookupTransform('/base_link', '/sick_back_left', rospy.Time(0))
        tf_back_right = listener.lookupTransform('/base_link', '/sick_back_right', rospy.Time(0))
        tf_back_middle = listener.lookupTransform('/base_link', '/sick_back_middle', rospy.Time(0))

        getMat(tf_front)
        getMat(tf_back_left)
        getMat(tf_back_right)
        getMat(tf_back_middle)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue

    front_sub = message_filters.Subscriber('/front/sick_safetyscanners/scan', LaserScan)
    left_sub = message_filters.Subscriber('/back_left/sick_safetyscanners/scan', LaserScan)
    right_sub = message_filters.Subscriber('/back_right/sick_safetyscanners/scan', LaserScan)
    middle_sub = message_filters.Subscriber('/back_middle/scan', LaserScan)

    ts = message_filters.ApproximateTimeSynchronizer([front_sub, left_sub, right_sub, middle_sub], 1, 1)
    ts.registerCallback(callback)
    
    rospy.spin()
