# 音乐风格分类

#### 介绍
keras+tensorflow框架，构建音乐风格分类的CNN模型。
公开数据集为GTZAN。

#### 分数
##### 不分时30s acc：0.785
```
             precision    recall  f1-score   support

      blues       0.80      0.75      0.77        16        
  classical       0.95      0.95      0.95        20    
    country       0.75      0.86      0.80        21      
      disco       0.70      0.79      0.74        29        
     hiphop       0.84      0.76      0.80        21      
       jazz       0.96      0.88      0.92        25        
      metal       0.71      0.77      0.74        13       
        pop       0.76      0.76      0.76        17         
     reggae       0.72      0.81      0.76        16      
       rock       0.65      0.50      0.56        22  
avg / total       0.79      0.79      0.78       200
```
##### 分时3s acc：0.85
```
             precision    recall  f1-score   support

      blues       0.89      0.90      0.89       177      
  classical       0.97      0.98      0.97       183  
    country       0.75      0.81      0.78       176    
      disco       0.78      0.84      0.81       177      
     hiphop       0.90      0.76      0.83       170     
       jazz       0.89      0.92      0.91       172      
      metal       0.87      0.96      0.91       190      
        pop       0.85      0.82      0.84       171        
     reggae       0.86      0.82      0.84       188     
       rock       0.77      0.69      0.73       191
avg / total       0.85      0.85      0.85      1795
```

#### 项目仅可用于学术交流
