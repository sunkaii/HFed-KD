import time

class timerUT():
    def __init__(self, start_time=None, end_time=None):
        self.start_time = start_time
        self.end_time = end_time
        self.stamp = []
        self.segment = []
        self.start()
    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def duration(self, round_=2):
        if self.end_time == None or self.start_time == None:
            print('Timer not end yet')
            return
        return round((self.end_time - self.start_time), round_)
    
    def addStamp(self, info=None):
        if self.end_time != None:
            print("Timer has end")
            return
        #self.stamp.append([time.time(), info])
        self.stamp.append({"time":time.time(), "info":info})

    def checkStamp(self, idx=None):
        if idx and idx < len(self.stamp):
            print(f"{self.stamp[idx]['info']} at {round(self.stamp[idx]['time'], 2)}")
        else:
            for idx in range(len(self.stamp)):
                print(f"{self.stamp[idx]['info']} at {round(self.stamp[idx]['time'], 2)}")

    def segmentStart(self, info=None):
        self.segment.append({"start":time.time(), "end":None, "info":info})
    def segmentEnd(self):
        self.segment[-1]["end"] = time.time()
    def segmentDuration(self, idx):
        return round(self.segment[idx]["end"] - self.segment[idx]["start"], 2)