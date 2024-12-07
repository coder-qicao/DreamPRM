import sys
sys.path.append("/home/q9cao/python_project/multimodal_reasoning")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
path = "/home/q9cao/python_project/multimodal_reasoning"
os.chdir(path)

from inference.inference import inference

inference(dataset='MathVista', part='testmini', CoT_id=range(1,8), do_sample=True, temperature=0.3)
