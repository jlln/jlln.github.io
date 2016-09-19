---
layout: post
title:  "Animal Rescue Kaggle Competition Part Two: XGBoost Model"
date:   2016-09-19 14:03:05 +0800
categories: kaggle,xgboost,modelling,classification
---

Kaggle recently ran a competition with the goal of predicting outcomes for animals in a shelter. In [part one]({% post_url 2016-09-19-animal-rescue-one %}), I posted my initial exploratory analysis of the data. Here, I will post the model that I developed to meet the objective of the competition. 

In this model I used calculated features (eg the proportion of animals with a given name that were adopted) in order to reduce dimensionality. This is often not a good idea, since it ignores any possible interaction effects of the consumed feature. However, in this instance it worked well. Using this approach does raise difficulties when conducting cross-validation, due to information leakage. In order to overcome this issue I wrote a custom cross validation function. 

This model produced a multiclass loss score of 0.69712 on the hidden testing dataset, putting it in the top 3% of competition entries.
{% include SimpleModels.html %}




