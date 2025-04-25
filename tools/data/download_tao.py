import openxlab
openxlab.login(ak="5l6kar3oaalk9a2rpnwv", sk="g6ldlmkqgpb2xro9dmxdb2gxgvzax7zvdrk0a5nb") #进行登录，输入对应的AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/TAO') #数据集信息及文件列表查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/TAO', target_path='/data/jiahaoguo/datasets/TAO/')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/TAO',source_path='/README.md', target_path='/data/jiahaoguo/datasets') #数据集文件下载