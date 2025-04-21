import logging
from tao import TaoEval
# 获取根日志记录器
root_logger = logging.getLogger()
# 修改日志级别为INFO
root_logger.setLevel(logging.INFO)
# TAO uses logging to print results. Make sure logging is set to show INFO
# messages, or you won't see any evaluation results.
tao_eval = TaoEval('/data/jiahaoguo/dataset/XS-VIDv2/annotations/jsonv2/test_segment.json',
                   "/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/track/run21/tracks.json", 
                   )
tao_eval.run()
tao_eval.print_results()