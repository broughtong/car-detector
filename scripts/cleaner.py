import rosbag
import os

datasetPath = "../data/rosbags"
outPath = "../data/rosbags_filtered"

def filterBag(filename, newfolder, newfilename):
    print(filename, newfolder, newfilename)
    os.makedirs(newfolder, exist_ok=True)
    with rosbag.Bag(newfilename, 'w') as newbag:
        with rosbag.Bag(filename, 'r') as bag:
            tss = []
            msgs = []
            ctr = 0
            for (topic, msg, ts) in bag.read_messages():
                if topic == "/tf_static":
                    newbag.write(topic, msg, ts)

                if topic != "/tf":
                    newbag.write(topic, msg, ts)

                else:
                    if msg.transforms[0].header.frame_id == "map":
                        continue
                    ctr += 1
                    if ctr < 30:
                        continue

                    #check for duplicates
                    stamp = msg.transforms[0].header.stamp
                    if stamp in tss:
                        idx = tss.index(stamp)
                        """
                        #uncomment to see duplicated tf messages
                        print("==dup==")
                        print(msgs[idx])
                        print(msg)
                        print(msgs[idx] == msg)
                        """
                        print("Duplicated message filtered!")
                        if msgs[idx] != msg:
                            print("Warning, duplicate message not exact match")
                        continue

                    tss.append(stamp)
                    msgs.append(msg)

                    #now write new bag
                    newbag.write(topic, msg, ts)

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".bag" in filename:
                path = datasetPath
                folder = files[0][len(path)+1:]
                oldfn = os.path.join(path, folder, filename)
                newfn = os.path.join(outPath, folder, filename)
                if "4" not in folder:
                    continue
                filterBag(oldfn, os.path.join(outPath, folder), newfn)