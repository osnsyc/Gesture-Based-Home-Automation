from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor

def initRKNN(rknnModel="model/rtmpose_hand_1x3x256x256.rknn", 
            id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        raise RuntimeError("Fail to load the first model.")
    
    core_masks = {
        0: RKNNLite.NPU_CORE_0,
        1: RKNNLite.NPU_CORE_1,
        2: RKNNLite.NPU_CORE_2,
        -1: RKNNLite.NPU_CORE_0_1_2
    }
    core_mask = core_masks.get(id, RKNNLite.NPU_CORE_0_1_2)
    
    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError("Fail to init the runtime of the first model.")

    print(rknnModel,"\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="model/rtmpose_hand_1x3x256x256.rknn", 
            TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_lite = initRKNN(rknnModel, i % 3)
        rknn_list.append(rknn_lite)
    return rknn_list


class PosePoolExecutor:
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame, box):
        rknn_lite = self.rknnPool[self.num % self.TPEs]
        self.queue.put(self.pool.submit(self.func, rknn_lite, frame, box))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()