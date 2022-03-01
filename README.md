# PSZT-U_Klasyfikator-sekwencji-DNA

## Summary
<p align="justify">
<i>Standard</i> parameters of the random forest algorithm proved to give good results, and are therefore a good starting point for further parameter tuning. 
</p>

<p align="justify">
On the basis of the obtained results it was possible to show that an increase in the number of trees in the random forest, can positively affect the quality of 
classification (up to a certain point, at which the model overfitting occurs). At the same time we observe that for the investigated problems the optimal, in terms 
of classification quality, number of trees is much lower than we expected. 
</p>

<p align="justify">
It is also confirmed that using all attributes to build a single tree allows the whole model to get the best results. This makes sense, as there will be trees in 
our forest capable of more accurate classification, and therefore the whole model will behave better. It is also worth noting here that the time required to build 
such a model is significantly larger, so for very large problems it may not make sense to use all attributes due to the excessive time required to train the model. 
</p>
  
<p align="justify">
Surprisingly, the size of the training set had no significant effect on the overall classification quality of the model. The only deviation here is the result for 
the spliceA set and the collection size of 90%, in which model overfitting most likely occurred. Lack of greater impact on the quality of classification is probably 
due to the specificity of the set, which is dominated by negative examples, which are about 4 times more numerous.
</p>
  
<p align="justify">
We were surprised by the time needed to run the above tests. The first results (which, due to an incorrectly designed test procedure, are not included in the 
report) were obtained after more than 13 hours of running the testing programme. The final results analysed and described in this document were also obtained after 
several hours of calculations. 
</p>

Above summary is part of final project documentation.
